import math
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from functools import reduce
from torch import nn
from typing import Annotated, Callable, Sequence

#####################
### Basics
#####################


def zero_init(layer: nn.Module) -> nn.Module:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def scale_for_cosine_sim(
    q: Annotated[torch.Tensor, "... d, float"],
    k: Annotated[torch.Tensor, "... d, float"],
    scale: Annotated[torch.Tensor, "... d, float"],
    eps: float = 1e-6,
) -> tuple[Annotated[torch.Tensor, "... d, float"], Annotated[torch.Tensor, "... d, float"]]:
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)

def linear_swiglu(
    x: Annotated[torch.Tensor, "... d_in, float"],
    weight: Annotated[torch.Tensor, "(2 d_out) d_in, float"],
    bias: Annotated[torch.Tensor, "(2 d_out), float"] | None = None,
) -> Annotated[torch.Tensor, "... d_out, float"]:
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)

class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(
        self, x: Annotated[torch.Tensor, "... d_in, float"],
    ) -> Annotated[torch.Tensor, "... d_out, float"]:
        return linear_swiglu(x, self.weight, self.bias)

def rms_norm(x: Annotated[torch.Tensor, "... d, float"], scale: Annotated[torch.Tensor, "... d, float"], eps: float) -> Annotated[torch.Tensor, "... d, float"]:
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class AdaRMSNorm(nn.Module):
    def __init__(self, d_model: int, d_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(d_cond, d_model, bias=False))

    def forward(
        self,
        x: Annotated[torch.Tensor, "... d_model, float"],
        cond: Annotated[torch.Tensor, "... d_cond, float"],
    ) -> Annotated[torch.Tensor, "... d_model, float"]:
        scale = self.linear(cond)
        if len(x.shape) > len(scale.shape):
            scale = scale.unsqueeze(1)
        return rms_norm(x, scale + 1, self.eps)

class RMSNorm(nn.Module):
    def __init__(self, shape: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x: Annotated[torch.Tensor, "... d, float"]) -> Annotated[torch.Tensor, "... d, float"]:
        return rms_norm(x, self.scale, self.eps)


class AdaLN(nn.Module):
    def __init__(
        self,
        shape: int | Sequence[int],
        cond_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.linear = zero_init(nn.Linear(cond_features, math.prod(self.shape) * 2, bias=True))

    def forward(self, x, cond, **kwargs):
        d = torch.promote_types(x.dtype, torch.float32)
        scale, shift = rearrange(self.linear(cond), "... (n d) -> n ... d", n=2)
        result = F.layer_norm(x.to(d), self.shape, weight=None, bias=None, eps=self.eps).to(dtype=x.dtype)
        result = result * (1 + scale) + shift
        return result

#####################
### Pos Utils
#####################

def make_grid_2d(b: int, h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
    h_edges = torch.linspace(-1, 1, h + 1, device=device, dtype=torch.float32)
    w_edges = torch.linspace(-1, 1, w + 1, device=device, dtype=torch.float32)
    h_pos = (h_edges[:-1] + h_edges[1:]) / 2
    w_pos = (w_edges[:-1] + w_edges[1:]) / 2
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    grid = repeat(grid, "h w c -> b h w c", b=b)
    return grid

def make_grid_3d(b: int, t: int, h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
    grid_2d = make_grid_2d(b, h, w, device=device)
    grid_2d = repeat(grid_2d, "b h w c -> b t h w c", t=t)
    t_pos = torch.arange(t, device=device, dtype=torch.float32)
    t_pos = repeat(t_pos, "t -> b t h w 1", b=b, h=h, w=w)
    return torch.cat([grid_2d, t_pos], dim=-1)

def make_freqs(
    d_head: int,
    nr_heads: int,
    min_freq: float,
    max_freq: float,
):
    d_axis = d_head // 8
    freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), nr_heads * d_axis + 1)[:-1])
    return rearrange(freqs, "(d h) -> d h", d=d_axis, h=nr_heads).mT.contiguous()

class RoPE2D(nn.Module):
    def __init__(
        self,
        d_head: int,
        nr_heads: int,
        min_freq: float = math.pi,
        max_freq: float = 10.0 * math.pi,
    ):
        super().__init__()
        for k in ["h", "w"]:
            self.register_buffer(f"freqs_{k}", make_freqs(d_head, nr_heads, min_freq, max_freq))

    def forward(self, pos: Annotated[torch.Tensor, "..., float"]) -> Annotated[torch.Tensor, "... nh dh, float"]:
        theta_h = repeat(pos[..., 0], "... -> ... 1 1") * self.freqs_h
        theta_w = repeat(pos[..., 1], "... -> ... 1 1") * self.freqs_w
        return torch.cat([theta_h, theta_w], dim=-1)

    def apply_emb(
        self,
        x: Annotated[torch.Tensor, "b nh l dh, float"],
        theta: Annotated[torch.Tensor, "b nh l dtheta, float"],
    ) -> Annotated[torch.Tensor, "b nh l dh, float"]:
        dtype = reduce(torch.promote_types, [x.dtype, theta.dtype, torch.float32])

        _, _, _, d_theta = theta.shape
        _, _, _, d = x.shape
        assert d_theta * 2 <= d, f"{theta.shape=} {x.shape=}"
        x1, x2, x3 = x[..., :d_theta], x[..., d_theta:2*d_theta], x[..., 2*d_theta:]
        x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
        cos_theta, sin_theta = theta.cos(), theta.sin()
        y1 = x1 * cos_theta - x2 * sin_theta
        y2 = x2 * cos_theta + x1 * sin_theta
        y1, y2 = y1.to(x.dtype), y2.to(x.dtype)
        return torch.cat([y1, y2, x3], dim=-1)

class RoPE3D(RoPE2D):
    def __init__(
        self,
        d_head: int,
        nr_heads: int,
        t_min_freq: float = 1.0,
        t_max_freq: float = 0.01,
        **kwargs,
    ):
        super().__init__(d_head, nr_heads, **kwargs)
        self.register_buffer("freqs_t", make_freqs(d_head, nr_heads, t_min_freq, t_max_freq))

    def forward(self, pos: Annotated[torch.Tensor, "..., float"]) -> Annotated[torch.Tensor, "... nh dh, float"]:
        theta_h = repeat(pos[..., 0], "... -> ... 1 1") * self.freqs_h
        theta_w = repeat(pos[..., 1], "... -> ... 1 1") * self.freqs_w
        theta_t = repeat(pos[..., 2], "... -> ... 1 1") * self.freqs_t
        return torch.cat([theta_t, theta_h, theta_w,], dim=-1)



#####################
### Layers and Blocks
#####################


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input: Annotated[torch.Tensor, "... d_in, float"]) -> Annotated[torch.Tensor, "... d_out, float"]:
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int = 0,
        exp_factor: int = 3,
    ):
        super().__init__()
        if d_cond > 0:
            self.norm = AdaRMSNorm(d_model, d_cond)
        else:
            self.norm = RMSNorm(d_model)
        d_ff = d_model * exp_factor
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))
    def forward(
        self,
        x: Annotated[torch.Tensor, "... d_model, float"],
        cond_norm: Annotated[torch.Tensor, "... d_cond, float"] | None = None,
    ) -> Annotated[torch.Tensor, "... d_model, float"]:
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return x + skip

class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 512
        self.in_norm = RMSNorm(512)
        self.blocks = nn.Sequential(FFN(512), FFN(512))
        self.out_norm = RMSNorm(512)

    def forward(self, x: Annotated[torch.Tensor, "... d, float"]) -> Annotated[torch.Tensor, "... d, float"]:
        x = self.in_norm(x)
        x = self.blocks(x)
        x = self.out_norm(x)
        return x

class SimpleProj(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_input: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.is_input = is_input

    def forward(self, x: Annotated[torch.Tensor, "... d_in, float"], pos: Annotated[torch.Tensor, "..., float"] | None = None, **kwargs) -> Annotated[torch.Tensor, "... d_out, float"] | tuple[Annotated[torch.Tensor, "... d_out, float"], Annotated[torch.Tensor, "..., float"]]:
        result = self.proj(x)
        if self.is_input:
            return result, pos
        return result

class SelfAttnLayer(nn.Module):
    def __init__(
        self,
        d_head: int,
        nr_heads: int,
        pos_emb_init: Callable[[], nn.Module],
        d_cond: int = 0,
    ):
        super().__init__()
        width = d_head * nr_heads
        if d_cond > 0:
            self.norm = AdaRMSNorm(width, d_cond)
        else:
            self.norm = RMSNorm(width)
        self.nr_heads = nr_heads
        self.pos_emb = pos_emb_init()
        self.qkv_proj = nn.Linear(width, width * 3, bias=False)
        self.scale = nn.Parameter(torch.full([nr_heads,], 10.0))
        self.out_proj = zero_init(nn.Linear(width, width, bias=False))
    
    def forward(
        self,
        x: Annotated[torch.Tensor, "b l d, float"],
        pos: Annotated[torch.Tensor, "b l d_pos, float"] | None = None,
        cond_norm: Annotated[torch.Tensor, "b l d_cond | b d_cond, float"] | None = None,
        **kwargs,
    ) -> Annotated[torch.Tensor, "b l d, float"]:
        skip = x
        B, *DIMS, _ = x.shape
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = rearrange(x, "b ... d -> b (...) d")
        qkv = self.qkv_proj(x)
        q,k,v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", nh=self.nr_heads, qkv=3)
        q, k = scale_for_cosine_sim(q, k, repeat(self.scale, "nh -> 1 nh 1 1"), eps=1e-6)
        if pos is not None:
            pos = rearrange(pos, "b ... d -> b (...) d")
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0,)
        x = rearrange(x, "b nh l dh -> b l (nh dh)")
        x = x.view(B, *DIMS, x.size(-1))
        x = self.out_proj(x)
        return x + skip

class CrossAttnLayer(nn.Module):
    def __init__(
        self,
        d_head: int,
        nr_heads: int,
        d_cross: int,
        pos_emb_init: Callable[[], nn.Module],
        d_cond: int = 0,
    ):
        super().__init__()
        width = d_head * nr_heads
        if d_cond > 0:
            self.norm = AdaRMSNorm(width, d_cond)
        else:
            self.norm = RMSNorm(width)
        self.norm_cross = RMSNorm(d_cross)
        self.nr_heads = nr_heads
        self.pos_emb = pos_emb_init()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.kv_proj = nn.Linear(d_cross, width * 2, bias=False)
        self.scale = nn.Parameter(torch.full([nr_heads,], 10.0))
        self.out_proj = zero_init(nn.Linear(width, width, bias=False))
    
    def forward(
        self,
        x: Annotated[torch.Tensor, "b l d, float"],
        x_cross: Annotated[torch.Tensor, "b l_cross d_cross, float"],
        pos: Annotated[torch.Tensor, "b l d_pos, float"] | None = None,
        pos_cross: Annotated[torch.Tensor, "b l_cross d_pos_cross, float"] | None = None,
        cond_norm: Annotated[torch.Tensor, "b l d_cond | b d_cond, float"] | None = None,
        **kwargs,
    ) -> Annotated[torch.Tensor, "b l d, float"]:
        skip = x
        B, *DIMS, _ = x.shape
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = rearrange(x, "b ... d -> b (...) d")
        x_cross = self.norm_cross(x_cross)
        x_cross = rearrange(x_cross, "b ... d -> b (...) d")
        q = self.q_proj(x)
        kv = self.kv_proj(x_cross)
        q = rearrange(q, "b l_q (nh dh) -> b nh l_q dh", nh=self.nr_heads)
        k, v = rearrange(kv, "b l_kv (kv nh dh) -> kv b nh l_kv dh", nh=self.nr_heads, kv=2)
        q, k = scale_for_cosine_sim(q, k, repeat(self.scale, "nh -> 1 nh 1 1"), eps=1e-6)
        if pos_cross is not None and pos is not None:
            pos = rearrange(pos, "b ... d -> b (...) d")
            pos_cross = rearrange(pos_cross, "b ... d -> b (...) d")
            theta = self.pos_emb(pos)
            theta_cross = self.pos_emb(pos_cross)
            theta = theta.movedim(-2, -3)
            theta_cross = theta_cross.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta_cross)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0,)
        x = rearrange(x, "b nh l dh -> b l (nh dh)")
        x = self.out_proj(x)
        x = x.view(B, *DIMS, x.size(-1))
        x = x + skip
        return x

class AttnBlock(nn.Module):
    def __init__(
        self,
        width: int,
        d_cond_norm: int = 0,
        self_attn_params: dict = {},
        cross_attn_params: dict | None = None,
    ):
        super().__init__()
        d_head = self_attn_params.get("d_head", 64)
        assert width % d_head == 0, f"{width=} must be divisible by {d_head=}"
        nr_heads = width // d_head
        self.self_attn = SelfAttnLayer(**(
            self_attn_params | {"d_head": d_head, "nr_heads": width // d_head, "d_cond": d_cond_norm}
        ))
        self.cross_attn = None\
            if cross_attn_params is None\
            else CrossAttnLayer(
                d_head,
                nr_heads,
                **(cross_attn_params | {"d_cond": d_cond_norm}),
            )
        self.ff = FFN(d_model=width, d_cond=d_cond_norm,)
    def forward(
        self,
        x: Annotated[torch.Tensor, "b l d, float"],
        pos: Annotated[torch.Tensor, "b l d_pos, float"] | None = None,
        x_cross: Annotated[torch.Tensor, "b l_cross d_cross, float"] | None = None,
        pos_cross: Annotated[torch.Tensor, "b l_cross d_pos_cross, float"] | None = None,
        cond_norm: Annotated[torch.Tensor, "b l d_cond | b d_cond, float"] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.self_attn(x, pos=pos, cond_norm=cond_norm, **kwargs)
        if self.cross_attn is not None:
            x = self.cross_attn(x, pos=pos, x_cross=x_cross, pos_cross=pos_cross, cond_norm=cond_norm, **kwargs)
        x = self.ff(x, cond_norm=cond_norm)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        in_proj: nn.Module,
        out_proj: nn.Module,
        d_cond_norm: int = 0,
        self_attn_params: dict = {},
        cross_attn_params: dict | None = None,
    ):
        super().__init__()
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.layers = nn.ModuleList([AttnBlock(width, d_cond_norm, self_attn_params, cross_attn_params) for _ in range(depth)])
    
    def forward(self, x: torch.Tensor, pos : torch.Tensor | None, **kwargs) -> torch.Tensor:
        x, pos = self.in_proj(x, pos=pos, **kwargs)
        for layer in self.layers:
            x = layer(x, pos=pos, **kwargs)
        x = self.out_proj(x)
        return x
