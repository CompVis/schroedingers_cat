import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import nn
from typing import Annotated

from .backbone import make_grid_3d, Transformer, SimpleProj, RoPE3D
from .dinov2 import MinDinoV2Reg

class TrackIdentityTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.nr_tracks = 64
        self.table = nn.Embedding(64, 512)

    def forward(self, batch_size: int, nr_tracks: int, device: torch.device):
        assert 1 <= nr_tracks <= 64
        track_idx = torch.arange(nr_tracks, device=device)
        return repeat(self.table(track_idx), "N D -> B N D", B=batch_size)

class ResidualWrapper(nn.Sequential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + x

class TrackInputMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3344, 1024, bias=False),
            *[
                ResidualWrapper(
                    nn.RMSNorm(1024),
                    nn.Linear(1024, 1024, bias=False),
                    nn.SiLU(),
                    nn.Linear(1024, 1024, bias=False),
                )
                for _ in range(2)
            ],
        )
        self.query_token = nn.Parameter(torch.empty(1, 1, 2048))
        torch.nn.init.trunc_normal_(self.query_token)
        self.coeffs = nn.Parameter(
            (2 * torch.pi * (torch.arange(512) + 1))[None, None, None, :],
            requires_grad=False,
        )
    
    def forward(
        self,
        tracks: torch.Tensor,        # B T N 2, float
        pos: torch.Tensor | None,
        is_query: torch.Tensor,   # B T N, bool
        pos_in_cond: torch.Tensor,   # B N 2, float
        aux_feats: torch.Tensor,  # B D_aux H W, float
        track_id: torch.Tensor | None,
        static_cam_emb: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # B L D_model
        _, T, N, _ = tracks.shape
        pre_emb_ = tracks[..., None] * self.coeffs.to(tracks)
        fourier_emb: torch.Tensor = rearrange(
            torch.stack([torch.sin(pre_emb_), torch.cos(pre_emb_)], dim=0,), 
            "sc b t n d_flow n_coeff -> b t n (d_flow sc n_coeff)",
        )
        track_emb = torch.where(is_query[..., None], self.query_token, fourier_emb)
        aux_feats_ = rearrange(
            F.grid_sample(
                aux_feats,
                repeat(
                    pos_in_cond.mul(2).sub(1),
                    "B N D_flow -> B N 1 D_flow",
                ),
                align_corners=False,
                mode="bilinear",
                padding_mode="border",
            ),
            "b d_aux n 1 -> b 1 n d_aux",
        )
        aux_feats_ = repeat(aux_feats_, "b 1 ... -> b t ...", t=T)
        x = torch.cat(
            [
                track_emb,
                repeat(track_id, "B N D -> B T N D", T=T),
                aux_feats_,
                repeat(static_cam_emb, "B D -> B T N D", T=T, N=N),
            ],
            dim=-1,
        )
        return self.mlp(x), pos


class TrajDistribEncoder(nn.Module):

    def __init__(
        self,
        backbone: Transformer,
        image_encoder: MinDinoV2Reg,
        track_identity_table: TrackIdentityTable,
    ):
        super().__init__()
        self.backbone = backbone
        self.image_encoder = image_encoder
        self.track_identity_table = track_identity_table
        self.static_camera_emb = nn.Embedding(2, 16)

    def precompute(
        self,
        cond_frame: Annotated[torch.Tensor, "B C H W, float, [-1, 1]"],
        tracks: Annotated[torch.Tensor, "B T N D, float, [0, 1]"],
    ) -> dict[str, torch.Tensor]:
        B, T, N, D = tracks.shape
        assert D == 2
        device = tracks.device
        img_feats: Annotated[torch.Tensor, "B H' W' D_img"] = self.image_encoder(cond_frame).clone()
        _, H, W, _ = img_feats.shape
        B_idx = torch.arange(start=0, end=B, device=device)
        pos_in_cond = tracks[B_idx, 0].clone()
        rope_pos_xy = pos_in_cond * 2 - 1
        rope_pos_xy = repeat(rope_pos_xy, f"B N D -> B T N D", T=T)
        rope_pos_t = torch.arange(T, dtype=tracks.dtype, device=device)
        rope_pos_t = repeat(rope_pos_t, f"T -> B T N", B=B, N=N,)
        rope_pos = torch.stack([
                rope_pos_xy[..., 1],
                rope_pos_xy[..., 0],
                rope_pos_t,
            ],
            dim=-1,
        )
        
        frame_pos = make_grid_3d(b=B, t=T, h=H, w=W, device=device)
        frame_pos = frame_pos[B_idx, 0]
        return {
            "pos_in_cond": pos_in_cond,
            "aux_feats": rearrange(img_feats, "b ... d -> b d ..."),
            "pos": rope_pos,
            "x_cross": img_feats,
            "pos_cross": frame_pos,
        }
    
    def forward(
        self,
        tracks: Annotated[torch.Tensor, "B T N D, float, [0, 1]"],
        is_query: Annotated[torch.Tensor, "B T N, bool"],
        is_camera_static: Annotated[torch.Tensor, "B, bool"],
        precomputed_kwargs: dict[str, torch.Tensor],
        **kwargs,
    ):
        B, T, N = is_query.shape
        track_id = self.track_identity_table(B, N, tracks.device)
        static_cam_emb = self.static_camera_emb(is_camera_static.int())
        result = self.backbone(
            tracks,
            is_query=is_query,
            track_id=track_id,
            static_cam_emb=static_cam_emb,
            **precomputed_kwargs,
        )
        result = torch.tanh(result)
        if self.training:
            result = result + torch.randn_like(result) * 1e-5
        return result

def make_encoder() -> TrajDistribEncoder:
    return TrajDistribEncoder(
        backbone=Transformer(
            width=1024,
            depth=24,
            in_proj=TrackInputMLP(),
            out_proj=SimpleProj(1024, 64),
            self_attn_params={
                "d_head": 64,
                "pos_emb_init": lambda: RoPE3D(d_head=64, nr_heads=16),
            },
            cross_attn_params={
                "d_cross": 768,
                "pos_emb_init": lambda: RoPE3D(d_head=64, nr_heads=16),
            },
        ),
        track_identity_table=TrackIdentityTable(),
        image_encoder=MinDinoV2Reg(),
    )
