import torch

from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F
from typing import Annotated

from .backbone import make_grid_2d, Transformer, SimpleProj, RMSNorm, RoPE2D


def sample_perm(B: int, dim: int, nr_samples: int, device: torch.device):
    perm = torch.stack(
        [torch.randperm(dim, device=device)[:nr_samples] for _ in range(B)],
        dim=0,
    )
    return perm

class ResidualWrapper(nn.Sequential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + x

class GridEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 768
        self.mlp = nn.Sequential(
            nn.Linear(1600, 768, bias=False),
            *[
                ResidualWrapper(
                    RMSNorm(768),
                    nn.Linear(768, 768, bias=False),
                    nn.SiLU(),
                    nn.Linear(768, 768, bias=False),
                )
                for _ in range(2)
            ],
        )
        self.register_buffer(
            "coeffs",
            repeat(2 * torch.pi * (torch.arange(384, dtype=torch.float32) + 1), "C -> 1 1 1 C"),
        )
    def forward(
        self,
        grid: torch.Tensor,  # B N D
        aux_feats: torch.Tensor  # B N D
    ):
        pre_emb_ = grid * self.coeffs
        pre_emb = torch.stack([torch.sin(pre_emb_), torch.cos(pre_emb_)], dim=0,)
        fourier_emb = rearrange(pre_emb, "sc b n d_flow n_coeff -> b n (d_flow sc n_coeff)")
        emb = torch.cat([fourier_emb, aux_feats], dim=-1)
        emb = self.mlp(emb)
        return emb


class DensityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Transformer(
            width=768,
            depth=12,
            in_proj=SimpleProj(768, 768, is_input=True),
            out_proj=SimpleProj(768, 1),
            self_attn_params={
                "d_head": 64,
                "pos_emb_init": lambda: RoPE2D(d_head=64, nr_heads=12),
            },
        )
        self.grid_embedder = GridEmbedder()
        self.grid_size = 20
        self.sampling_kwargs = {}
        self.prepend_latent_proj = nn.Linear(64, 768, bias=False)
        self.out_of_bounds_token = nn.Parameter(torch.randn(768))

    def forward(
        self,
        latent: Annotated[torch.Tensor, "b t n d | b (n_global + (t n)) d"],
        **kwargs,
    ):
        flat_latent = rearrange(latent, "B TN D -> (B TN) D")
        BTN = flat_latent.shape[0]
        grid_pos = make_grid_2d(b=BTN, h=self.grid_size, w=self.grid_size, device=latent.device)
        grid = rearrange(grid_pos[..., [1, 0]], "BTN H W C -> BTN (H W) C 1")
        grid_pos = rearrange(grid_pos, "BTN H W C -> BTN (H W) C")
        zero_pos = torch.zeros(size=(BTN, 1, 2), device=grid_pos.device, dtype=grid_pos.dtype)
        grid_pos = torch.cat([zero_pos, grid_pos, zero_pos], dim=1)
        HW = grid.shape[1]
        aux_feats = repeat(flat_latent, "BTN D -> BTN HW D", HW=HW)
        grid_emb = self.grid_embedder(grid, aux_feats)
        seq = [
            repeat(self.prepend_latent_proj(flat_latent), "BTN D -> BTN 1 D"),
            grid_emb,
            repeat(self.out_of_bounds_token, "D -> BTN 1 D", BTN=BTN),
        ]
        seq = torch.cat(seq, dim=1,)
        return self.backbone(seq, pos=grid_pos)[:, 1:]

    def loss(
        self,
        x: Annotated[torch.Tensor, "b t n c"],
        latent: Annotated[torch.Tensor, "b t n d | b (n_global + (t n)) d"],
        *args,
        **kwargs,
    ):
        B, T, N, _ = x.shape
        device = x.device
        nr_T_samples = self.sampling_kwargs.get("T", T)
        assert nr_T_samples > 0 and nr_T_samples <= T, f"{nr_T_samples=} {T=}"
        nr_N_samples = self.sampling_kwargs.get("N", N)
        assert nr_N_samples > 0 and nr_N_samples <= N, f"{nr_N_samples=} {N=}"
        B_idx = repeat(torch.arange(B, device=device), "B -> B 1 1")
        T_perm = rearrange(
            sample_perm(B=B*nr_N_samples, dim=T, nr_samples=nr_T_samples, device=device),
            "(B N) T -> B T N",
            B=B,
            N=nr_N_samples,
        )
        N_perm = rearrange(
            sample_perm(B=B*nr_T_samples, dim=N, nr_samples=nr_N_samples, device=device),
            "(B T) N -> B T N",
            B=B,
            T=nr_T_samples,
        )
        new_latent = latent[B_idx, T_perm, N_perm]
        new_x_gt = x[B_idx, T_perm, N_perm]

        pred = self.forward(
            latent=rearrange(new_latent, "B T N D -> B (T N) D"),
            *args,
            **kwargs,
        )
        logits = pred.squeeze(-1)

        grid_coords = (new_x_gt * (self.grid_size - 1)).round().long()
        in_bounds = ((new_x_gt >= 0.0) & (new_x_gt <= 1.0)).all(dim=-1)
        target_idx = grid_coords[..., 0] * self.grid_size + grid_coords[..., 1]
        target_idx = torch.where(in_bounds, target_idx, self.grid_size ** 2).reshape(-1)
        return F.cross_entropy(logits, target_idx)
    
    def decode_latent(
        self,
        latent: Annotated[torch.Tensor, "b t n d | b (n_global + (t n)) d"],
        with_oob_tok: bool = False,
        split_bs: int=-1,
        *args, **kwargs,
    ):
        B, T, N, C = latent.shape
        M = B * T * N

        if split_bs > 0 and M > split_bs:
            latent_pts = rearrange(latent, "B T N D -> (B T N) 1 D")
            preds = []
            for start in range(0, M, split_bs):
                end = min(start + split_bs, M)
                latent_chunk = latent_pts[start:end]
                pred_chunk = self.forward(
                    latent=latent_chunk.clone(),
                    *args, **kwargs,
                ).clone()
                preds.append(pred_chunk)
            pred = torch.cat(preds, dim=0)
        else:
            pred = self.forward(
                latent=rearrange(latent, "B T N D -> B (T N) D"),
                *args, **kwargs,
            )
        pred = F.softmax(pred.squeeze(-1), dim=-1).unsqueeze(-1)
        pred = rearrange(pred, "(B T N) L C -> B (T N) L C", B=B, T=T, N=N)
        oob_tok = pred[:, :, -1]
        grid = pred[:, :, :-1]
        grid = rearrange(grid, "B (T N) (H W) C -> B T N H W C", T=T, N=N, H=self.grid_size, W=self.grid_size)
        if with_oob_tok:
            return grid, oob_tok
        return grid

def make_density_decoder() -> DensityEstimator:
    return DensityEstimator()
