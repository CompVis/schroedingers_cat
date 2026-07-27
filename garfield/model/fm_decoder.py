
import torch
from einops import rearrange, repeat
from torch import nn
from typing import Annotated

from .backbone import zero_init, AdaLN, MappingNetwork, FourierFeatures, Transformer, SimpleProj, RoPE3D


class FM(nn.Module):
    
    def __init__(
        self,
        backbone: nn.Module,
        time_mapping: MappingNetwork,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        self.time_emb = FourierFeatures(1, time_mapping.width)
        self.time_in_proj = nn.Linear(time_mapping.width, time_mapping.width, bias=False)
        self.time_mapping = time_mapping

        self.latent_dropout_token = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

    def get_t_shape(self, data_dims: tuple[int, ...]) -> tuple[int, ...]:
        return (*data_dims[:-1], 1)
    
    def sample_train_t(self, data_dims: tuple[int, ...], device: torch.device) -> Annotated[torch.Tensor, "b ..., float"]:
        return torch.sigmoid(torch.randn(self.get_t_shape(data_dims), device=device))

    def get_sampling_t(self, data_dims: tuple[int, ...], steps: int, device: torch.device) -> Annotated[torch.Tensor, "(steps+1) b ..., float"]:
        B, T, N, _ = self.get_t_shape(data_dims)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
        return repeat(ts, "S -> S B T N 1", B=B, T=T, N=N)

    def sample_cond_mask(
        self, data_dims: tuple[int, ...], device: torch.device
    ) -> Annotated[torch.Tensor, "b ..., bool"]:
        return torch.rand(self.get_t_shape(data_dims), device=device) >= 0.1

    def get_cond(
        self,
        t: Annotated[torch.Tensor, "b ..., float"],
        latent: Annotated[torch.Tensor, "b ..., float"],
        **data_kwargs,
    ) -> dict[str, torch.Tensor]:
        return {
            "cond_norm": self.time_mapping(self.time_in_proj(self.time_emb(t))),
            "latent": latent,
        }

    def loss(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def decode_latent(
        self,
        x_0: Annotated[torch.Tensor, "b ... c, float"],
        latent: Annotated[torch.Tensor, "b t n d, float"],
        *args, **kwargs,
    ):
        sample = self.sample(x_0=x_0.clone(), latent=latent.clone(), *args, **kwargs).clone()
        return sample

    def predict_v(
        self,
        x_t: Annotated[torch.Tensor, "b ... c, float"],
        cond_dropout_mask: Annotated[torch.Tensor, "b ... , bool"],
        latent: Annotated[torch.Tensor, "b t n d, float"],
        **cond_kwargs,
    ) -> Annotated[torch.Tensor, "b ... c, float"]:
        masked_latent = torch.where(
            cond_dropout_mask,
            latent,
            repeat(self.latent_dropout_token, "d -> 1 1 1 d"),
        )
        return self.backbone(x=x_t, latent=masked_latent, **cond_kwargs)

    def forward(
        self,
        x: Annotated[torch.Tensor, "b ... c, float"],
        latent: Annotated[torch.Tensor, "b ... d, float"],
        **data_kwargs,
    ) -> dict[str, Annotated[torch.Tensor, "b, float"]]:
        x_0 = torch.randn_like(x)
        t = self.sample_train_t(x.shape, device=x.device)
        x_t = (1 - t) * x_0 + t * x
        cond_dict = self.get_cond(t=t, latent=latent, x=x, **data_kwargs)
        cond_dict = data_kwargs | cond_dict
        v_pred = self.predict_v(
            x_t=x_t,
            cond_dropout_mask=self.sample_cond_mask(x.shape, device=x.device),
            **cond_dict,
        )
        return (v_pred - (x - x_0)).square().mean(dim=list(range(1, x.ndim)))

    @torch.no_grad()
    def sample(self, x_0: Annotated[torch.Tensor, "b ... c, float"], steps: int = 50, **data_kwargs) -> Annotated[torch.Tensor, "b ... c, float"]:
        ts = self.get_sampling_t(x_0.shape, steps=steps, device=x_0.device)
        mask_keep = torch.ones(self.get_t_shape(x_0.shape), device=x_0.device, dtype=torch.bool)
        x_t = x_0
        for t_start, t_end in zip(ts[:-1], ts[1:]):
            v_pred = self.predict_v(
                x_t=x_t.clone(),
                cond_dropout_mask=mask_keep.clone(),
                **self.get_cond(t=t_start, x=x_0, **data_kwargs),
            )
            x_t = x_t + (t_end - t_start) * v_pred
        return x_t




# Pointwise


class HeadFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = AdaLN(1024, 1024)
        self.sequential = nn.Sequential(
            nn.Linear(1024, 3072, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(3072, 1024, bias=False),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cond_norm=None,
    ):
        skip = x
        x = self.norm(x, cond_norm)
        x = self.sequential(x)
        return x + skip

class PointwiseMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.cond_emb = nn.Linear(64, 1024, bias=False)
        self.latent_proj = nn.Linear(64, 1024)
        self.cond_norm_proj = nn.Linear(512, 1024)
        self.emb_mapping = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(approximate="tanh"),
            nn.Linear(1024, 1024),
            nn.GELU(approximate="tanh"),
            nn.Linear(1024, 1024),
        )
        self.layers = nn.ModuleList([HeadFFN() for _ in range(3)])
        self.in_proj = nn.Linear(2, 1024, bias=False)
        self.out_proj = nn.Linear(1024, 2, bias=False)
    
    def embed(
        self,
        latent: Annotated[torch.Tensor, "B ... D_latent"],
        cond_norm: Annotated[torch.Tensor, "B ... D_temb"],
    ):
        latent_emb = self.latent_proj(latent)
        t_emb = self.cond_norm_proj(cond_norm)
        cond_emb = latent_emb + t_emb
        cond_emb = self.emb_mapping(cond_emb)
        return cond_emb
    
    def forward(
        self,
        x: Annotated[torch.Tensor, "B ... D_out"],
        latent: Annotated[torch.Tensor, "B ... D_latent"]  = None,
        cond_norm: Annotated[torch.Tensor, "B ... D_temb"] = None,
        **kwargs
    ):
        skip = x
        x = self.in_proj(x)
        cond_emb = self.embed(latent, cond_norm)
        for layer in self.layers:
            x = layer(x, cond_norm=cond_emb)
        x = self.out_proj(x)
        x = skip + x
        return x

def make_point_decoder() -> FM:
    return FM(
        backbone=PointwiseMLP(),
        time_mapping=MappingNetwork(),
        latent_dim=64,
    )

# Full


class CondLatentMerge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(66, 1152, bias=False)

    def forward(
        self,
        x: Annotated[torch.Tensor, "... t n d_in"],
        pos: Annotated[torch.Tensor, "... t n"] | None,
        latent: Annotated[torch.Tensor, "... t n d_lat"],
        **kwargs,
    ) -> tuple[Annotated[torch.Tensor, "... t n d_out"], Annotated[torch.Tensor, "... t n"] | None]:
        x = torch.cat((x, latent), dim=-1)
        return self.proj(x), pos

class FullDecoder(FM):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def get_t_shape(self, data_dims: tuple[int, ...]) -> tuple[int, ...]:
        return (data_dims[0], *([1] * (len(data_dims) - 1)))
    
    def get_cond(
        self,
        t: Annotated[torch.Tensor, "b ..., float"],
        latent: Annotated[torch.Tensor, "b ..., float"],
        x: Annotated[torch.Tensor, "b ... c, float"],
        is_camera_static: Annotated[torch.Tensor, "b, bool"] = None,
        **data_kwargs,
    ) -> dict[str, torch.Tensor]:
        cond_dict = super().get_cond(t=t, latent=latent, x=x, **data_kwargs)
        cond_norm = cond_dict["cond_norm"]
        _, t, n, _ = cond_norm.shape
        return cond_dict
    
    def get_pos(
        self,
        x_t: Annotated[torch.Tensor, "b ... c"],
        B: int,
        T: int,
        N: int,
        **kwargs,
    ):
        pos_in_cond = torch.rand(B, N, 2, device=x_t.device)
        rope_pos_xy = pos_in_cond * 2 - 1
        rope_pos_xy = repeat(rope_pos_xy, f"B N D -> B T N D", T=T)
        rope_pos_t = torch.arange(T, dtype=x_t.dtype, device=x_t.device)
        rope_pos_t = repeat(rope_pos_t, f"T -> B T N", B=B, N=N,)
        pos = torch.stack([
            rope_pos_xy[..., 1],
            rope_pos_xy[..., 0],
            rope_pos_t,
        ], dim=-1)
        return pos, kwargs

    def predict_v(self, x_t: Annotated[torch.Tensor, "b ... c"], **kwargs,):
        B, T, N, _ = x_t.shape
        pos, kwargs = self.get_pos(x_t=x_t, B=B, T=T, N=N, **kwargs)
        return super().predict_v(x_t=x_t, pos=pos, **kwargs)

def make_full_decoder() -> FullDecoder:
    return FullDecoder(
        backbone=Transformer(
            width=1152,
            depth=28,
            in_proj=CondLatentMerge(),
            out_proj=SimpleProj(1152, 2),
            self_attn_params={
                "d_head": 64,
                "pos_emb_init": lambda: RoPE3D(d_head=64, nr_heads=18),
            },
            d_cond_norm=512,
        ),
        time_mapping=MappingNetwork(),
        latent_dim=64,
    )
