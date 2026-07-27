import torch
import os
import click
import cv2
import tempfile
import socket
import signal
import contextlib
import einops
import io
import gc
import ffmpeg  # pip install git+https://github.com/kkroening/ffmpeg-python
import time
import dataclasses
import re
import logging
import random

import numpy as np
import torchvision.transforms.v2 as TVT
import webdataset as wds
from torch.multiprocessing import get_context
from jaxtyping import Float
import torch.nn.functional as F

from torch import nn
from tqdm.auto import tqdm
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence
from torchvision.models import vision_transformer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

cv2.setNumThreads(1)

MP4_SUFFIX = "mp4"
SUFFIX = ".tar"  # output shard name suffix
DTYPE = torch.bfloat16
DEVICE = "cuda"
TAPNEXT_INPUT_SIZE = (256, 256)  # training resolution of TAPNext
COMPILE = False

possible_ckpt_path = ["./bootstapnext_ckpt.npz"]


# --- TapNext Model implementation (https://github.com/google-deepmind/tapnet) ---

_MAX_SQRT_GRADIENT = 1000.0


def safe_div(numerator, denominator):
    return torch.where(
        torch.abs(denominator) < 1e-5,
        numerator * 100000.0,
        numerator / denominator,
    )


class PScan(torch.autograd.Function):
    """Implements a parallel scan operation.

    Given A is (N, T, D) and X is (N, T, D), expands A and X in-place in O(T),
    and O(log(T)) if not core-bounded, so that:
      Y[:, 0] = Y_init
      Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    can be computed as:
      Y[:, t] = A[:, t] * Y_init + X[:, t]
    """

    @classmethod
    def expand(cls, weights, bias):
        if weights.size(1) == 1:
            return
        t_even = 2 * (weights.size(1) // 2)

        w_pairs = weights[:, :t_even].view(weights.size(0), t_even // 2, 2, -1)
        b_pairs = bias[:, :t_even].view(bias.size(0), t_even // 2, 2, -1)

        b_pairs[:, :, 1].add_(w_pairs[:, :, 1] * b_pairs[:, :, 0])
        w_pairs[:, :, 1].mul_(w_pairs[:, :, 0])

        PScan.expand(w_pairs[:, :, 1], b_pairs[:, :, 1])

        b_pairs[:, 1:, 0].add_(w_pairs[:, 1:, 0] * b_pairs[:, :-1, 1])
        w_pairs[:, 1:, 0].mul_(w_pairs[:, :-1, 1])

        if t_even < weights.size(1):
            bias[:, -1].add_(weights[:, -1] * bias[:, -2])
            weights[:, -1].mul_(weights[:, -2])

    @classmethod
    def accrev(cls, tensor):
        if tensor.size(1) == 1:
            return
        t_even = 2 * (tensor.size(1) // 2)

        pairs = tensor[:, -t_even:].view(tensor.size(0), t_even // 2, 2, -1)

        pairs[:, :, 0].add_(pairs[:, :, 1])
        PScan.accrev(pairs[:, :, 0])
        pairs[:, :-1, 1].add_(pairs[:, 1:, 0])

        if t_even < tensor.size(1):
            tensor[:, 0].add_(tensor[:, 1])

    @classmethod
    def forward(cls, ctx, weights, bias, y_init):
        ctx.weights_orig = weights.clone()
        ctx.y_init_expanded = y_init[:, None, :].clone()
        ctx.weights_expanded = weights.clone()
        ctx.bias_expanded = bias.clone()

        PScan.expand(ctx.weights_expanded, ctx.bias_expanded)
        output = ctx.weights_expanded * ctx.y_init_expanded + ctx.bias_expanded
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_input_wrt_output = grad_output * ctx.weights_expanded
        grad_accumulated = grad_input_wrt_output.clone()

        PScan.accrev(grad_accumulated)

        grad_weights = safe_div(ctx.y_init_expanded, ctx.weights_orig)
        grad_weights[:, 1:].add_(safe_div(ctx.bias_expanded[:, :-1], ctx.weights_expanded[:, 1:]))

        grad_bias = safe_div(grad_accumulated, ctx.weights_expanded)
        grad_y_init = grad_input_wrt_output.sum(dim=1)

        return grad_weights * grad_accumulated, grad_bias, grad_y_init


pscan = PScan.apply


class RMSNorm(nn.Module):
    """RMS Norm."""

    def __init__(
        self,
        width: int,
        eps: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.eps = eps

        # Parameters.
        self.scale = nn.Parameter(torch.empty([self.width], device=device, dtype=dtype))

    def forward(self, x):
        """Calls the RMSNorm."""
        var = torch.mean(torch.square(x), axis=-1, keepdims=True)
        normed_x = x * torch.rsqrt(var + self.eps)

        scale = torch.reshape(self.scale, [1 for _ in range(x.ndim - 1)] + [-1])

        return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(
        self,
        width: int,
        num_blocks: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.block_width = self.width // self.num_blocks

        # Parameters.
        self.w = nn.Parameter(
            torch.empty(
                [self.num_blocks, self.block_width, self.block_width],
                device=device,
                dtype=dtype,
            )
        )
        self.b = nn.Parameter(torch.empty([self.num_blocks, self.block_width], device=device, dtype=dtype))

    def forward(self, x):
        """Calls the BlockDiagonalLinear."""
        # Split x to blocks
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


def rnn_scan(x, a, h0, acc_dtype=torch.float32, use_linear_scan=True):
    """Runs the recurrence of a linear RNN.

    Uses linear scan when given 1 timestep and parallel scan when given >1
    timesteps.

    Args:
      x: The input sequence.
      a: The diagonal of the recurrence matrix `A`.
      h0: The initial hidden state.
      acc_dtype: The data type for the accumulation.
      use_linear_scan: Whether to use linear scan.

    Returns:
      The output of the linear recurrence.
    """
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim :]
    assert a.dtype == x.dtype
    assert type(a) is type(x)
    assert h0 is None or h0.dtype == acc_dtype

    if x.shape[1] == 1:
        # Using scan in sampling mode.
        if h0 is None:
            return x, x[:, 0].type(acc_dtype)
        else:
            y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
            return y.type(x.dtype), y[:, -1]
    else:
        if h0 is not None:
            h_t = h0
        else:
            h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)
        if use_linear_scan:
            y = torch.zeros_like(x)
            for t in range(x.shape[1]):
                h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
                y[:, t] = h_t.type(x.dtype)
        else:
            # Using parallel scan!
            y = pscan(a, x, h_t)
            h_t = y[:, -1]
    return y, h_t


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters and layers.
        self.a_param = nn.Parameter(torch.empty([self.width], device=device, dtype=dtype))
        self.input_gate = BlockDiagonalLinear(
            width=self.width,
            num_blocks=self.num_heads,
            w_init_variance_scale=w_init_variance_scale,
            device=device,
            dtype=dtype,
        )
        self.a_gate = BlockDiagonalLinear(
            width=self.width,
            num_blocks=self.num_heads,
            w_init_variance_scale=self.w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, cache=None, use_linear_scan=True):
        _, l, _ = x.shape
        segment_pos = torch.arange(l, device=x.device)
        if cache is not None:
            segment_pos += 1
        reset = segment_pos == 0

        # Gates for x and a.
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        # Compute the parameter `A` of the recurrence.
        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)
        # Gate the input.
        gated_x = x * gate_x
        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16.
        multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset[..., None] + ~reset[..., None] * multiplier
        normalized_x = gated_x * multiplier.type(x.dtype)

        y, last_h = rnn_scan(x=normalized_x, a=a, h0=cache, use_linear_scan=use_linear_scan)

        return y, last_h

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        width: int,
        device: str | torch.device | None = None,
    ):
        """Returns an empty initialized cache for the RG-LRU."""
        # RG-LRU cache always in float32.
        return torch.zeros((batch_size, width), dtype=torch.float32, device=device)


class CausalConv1D(nn.Module):
    """A 1D temporal convolution layer."""

    def __init__(
        self,
        width: int,
        temporal_width: int,
        w_init_variance_scale: float = 0.01,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters.
        self.w = nn.Parameter(torch.empty([self.temporal_width, self.width], device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty([width], device=device, dtype=dtype))

    def forward(self, x, cache=None):
        if cache is None:
            cache = torch.zeros(
                (x.shape[0], self.temporal_width - 1, x.shape[2]),
                dtype=x.dtype,
                device=x.device,
            )
        assert cache.shape[1] == (self.temporal_width - 1)
        x = torch.cat([cache, x], dim=1)
        one_step = x.shape[1] == self.temporal_width
        if one_step:
            y = (x * self.w.unsqueeze(0)).sum(1, keepdims=True) + self.b[None, None, :]
        else:
            y = F.conv1d(x.transpose(1, 2), self.w.t().unsqueeze(1), self.b, groups=x.shape[-1])
            y = y.transpose(1, 2).contiguous()
        new_cache = x[:, 1 - self.temporal_width :]
        return y, new_cache

    @classmethod
    def init_cache(
        cls, *, batch_size: int, width: int, dtype: torch.dtype, conv1d_temporal_width: int = 4, device=None
    ):
        """Returns an empty initialized cache for the Conv1D."""
        shape = (batch_size, conv1d_temporal_width - 1, width)
        return torch.zeros(shape, dtype=dtype, device=device)


class Einsum(nn.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    def __init__(
        self,
        w_shape: Sequence[int],
        b_shape: Sequence[int],
        eqn: str,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w_shape = tuple(w_shape)
        self.b_shape = tuple(b_shape)
        self.eqn = eqn
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters.
        self.w = nn.Parameter(torch.empty(self.w_shape, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty(self.b_shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the Einsum."""
        return torch.einsum(self.eqn, x, self.w) + self.b


class RecurrentBlockCache(NamedTuple):
    rg_lru_state: torch.Tensor  # "*b e"
    conv1d_state: torch.Tensor  # "*b w e"


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Returns the GELU activation function with the same approximation as JAX."""
    return F.gelu(x, approximate="tanh")


class RecurrentBlock(nn.Module):
    """A block that combines a linear layer, a 1D convolution, and an RG-LRU."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width or width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Layers.
        self.linear_y = nn.Linear(
            in_features=self.width,
            out_features=self.lru_width,
            device=device,
            dtype=dtype,
        )
        self.linear_x = nn.Linear(
            in_features=self.width,
            out_features=self.lru_width,
            device=device,
            dtype=dtype,
        )
        self.linear_out = nn.Linear(
            in_features=self.lru_width,
            out_features=self.width,
            device=device,
            dtype=dtype,
        )
        self.conv_1d = CausalConv1D(
            width=self.lru_width,
            temporal_width=self.conv1d_temporal_width,
            device=device,
            dtype=dtype,
        )
        self.rg_lru = RGLRU(
            width=self.lru_width,
            num_heads=self.num_heads,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, cache: RecurrentBlockCache | None = None, use_linear_scan=True):
        y = self.linear_y(x)
        y = gelu(y)
        x = self.linear_x(x)
        x, conv1d_state = self.conv_1d(
            x=x,
            cache=None if cache is None else cache.conv1d_state,
        )
        x, rg_lru_state = self.rg_lru(
            x=x,
            cache=None if cache is None else cache.rg_lru_state,
            use_linear_scan=use_linear_scan,
        )

        # Join branches.
        x = x * y
        x = self.linear_out(x)

        return x, RecurrentBlockCache(
            conv1d_state=conv1d_state,
            rg_lru_state=rg_lru_state,
        )

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        lru_width: int,
        dtype: torch.dtype,
        conv1d_temporal_width: int = 4,
        device: str | torch.device | None = None,
    ) -> RecurrentBlockCache:
        """Initializes an empty RG-LRU and Conv1D cache for the block."""
        return RecurrentBlockCache(
            rg_lru_state=RGLRU.init_cache(
                batch_size=batch_size,
                width=lru_width,
                device=device,
            ),
            conv1d_state=CausalConv1D.init_cache(
                batch_size=batch_size,
                width=lru_width,
                dtype=dtype,
                conv1d_temporal_width=conv1d_temporal_width,
                device=device,
            ),
        )


class MLPBlock(nn.Module):
    """A block that implements a feed-forward network with a GELU activation."""

    def __init__(
        self,
        width: int,
        expanded_width: int,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.expanded_width = expanded_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Layers.
        self.ffw_up = Einsum(
            w_shape=(2, self.width, self.expanded_width),
            b_shape=(2, 1, 1, self.expanded_width),
            eqn="...td,cdD->c...tD",
            device=device,
            dtype=dtype,
        )
        self.ffw_down = nn.Linear(
            in_features=self.expanded_width,
            out_features=self.width,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        out = self.ffw_up(x)
        gate_value = gelu(out[0])
        activations = gate_value * out[1]
        return self.ffw_down(activations)


class ResidualBlock(nn.Module):
    """Griffin and Hawk's residual block."""

    def __init__(
        self,
        width: int,
        mlp_expanded_width: int,
        num_heads: int,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Sub-blocks and layers.
        self.temporal_pre_norm = RMSNorm(width=self.width, device=device, dtype=dtype)

        self.recurrent_block = RecurrentBlock(
            width=self.width,
            num_heads=self.num_heads,
            lru_width=self.lru_width,
            conv1d_temporal_width=self.conv1d_temporal_width,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

        self.channel_pre_norm = RMSNorm(
            width=width,
            device=device,
            dtype=dtype,
        )
        self.mlp_block = MLPBlock(
            width=self.width,
            expanded_width=self.mlp_expanded_width,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, cache: RecurrentBlockCache | None = None, use_linear_scan=True):
        raw_x = x
        inputs_normalized = self.temporal_pre_norm(raw_x)
        x, cache = self.recurrent_block(inputs_normalized, cache, use_linear_scan)
        residual = x + raw_x
        x = self.channel_pre_norm(residual)
        x = self.mlp_block(x)
        x = x + residual
        return x, cache

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        width: int,
        dtype: torch.dtype,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
        device: str | torch.device | None = None,
    ) -> RecurrentBlockCache:
        """Initializes an empty cache for the block."""
        return RecurrentBlock.init_cache(
            batch_size=batch_size,
            lru_width=lru_width or width,
            dtype=dtype,
            conv1d_temporal_width=conv1d_temporal_width,
            device=device,
        )


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=np.float32):
    """Follows the MoCo v3 logic."""
    y, x = np.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = np.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = np.einsum("m,d->md", y.flatten(), omega)
    x = np.einsum("m,d->md", x.flatten(), omega)
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    return np.asarray(pe, dtype)[None, :, :]


class TRecViTBlock(nn.Module):
    """A block proposed by https://arxiv.org/abs/2412.14294."""

    def __init__(self, depth, width, num_heads, lru_width, dtype, device):
        super().__init__()
        self.ssm_block = ResidualBlock(
            width=width,
            mlp_expanded_width=width * 4,
            num_heads=num_heads,
            lru_width=lru_width,
            final_w_init_variance_scale=2.0 / depth,
            dtype=dtype,
            device=device,
        )
        self.vit_block = vision_transformer.EncoderBlock(
            num_heads=num_heads,
            mlp_dim=width * 4,
            hidden_dim=width,
            attention_dropout=0.0,
            dropout=0.0,
        )

    def forward(self, x, cache=None, use_linear_scan=True):
        b, t, n, _ = x.shape
        x = einops.rearrange(x, "b t n c -> (b n) t c")
        x, ssm_cache = self.ssm_block(x, cache, use_linear_scan=use_linear_scan)
        x = einops.rearrange(x, "(b n) t c -> (b t) n c", b=b, n=n)
        x = self.vit_block(x)
        x = einops.rearrange(x, "(b t) n c -> b t n c", b=b, t=t)
        return x, ssm_cache


@dataclasses.dataclass
class TAPNextTrackingState:
    """State for TAPNext."""

    step: int
    query_points: torch.Tensor  # Float["*B Q 3"]
    hidden_state: list[RecurrentBlockCache] = None


class TAPNext(nn.Module):
    """TAPNext implementation in pytorch."""

    def __init__(
        self,
        image_size,
        width=768,
        patch_size=(8, 8),
        num_heads=12,
        lru_width=768,
        depth=12,
        use_checkpointing=False,
    ):
        super().__init__()
        self.width = width
        self.patch_size = patch_size
        self.use_checkpointing = use_checkpointing
        self.image_size = image_size

        self.lin_proj = nn.Conv2d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                TRecViTBlock(
                    depth=depth,
                    width=width,
                    num_heads=num_heads,
                    lru_width=lru_width,
                    dtype=torch.float32,
                    device="cuda",
                )
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.width)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, 1, self.width)), requires_grad=True)
        self.unknown_token = nn.Parameter(torch.zeros((1, 1, self.width)), requires_grad=True)
        self.point_query_token = nn.Parameter(torch.zeros((1, 1, 1, self.width)), requires_grad=True)
        h = self.image_size[0] // self.patch_size[0]
        w = self.image_size[1] // self.patch_size[1]
        c = self.width
        self.image_pos_emb = nn.Parameter(torch.zeros((1, h * w, c)), requires_grad=True)
        self.register_buffer(
            "query_pos_embed",
            torch.tensor(posemb_sincos_2d(self.image_size[0], self.image_size[1], c)),
        )
        self.visible_head = nn.Sequential(
            nn.Linear(width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.coordinate_head = nn.Sequential(
            nn.Linear(width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
        )

    def embed_queries(self, timesteps, query_points):
        b, q, _ = query_points.shape
        t = timesteps
        c = self.width
        tiled_point_query_tokens = self.point_query_token.repeat(b, 1, q, 1)
        mask_tokens = self.mask_token.repeat(b, t, q, 1)
        unknown_tokens = self.unknown_token.repeat(b, t, q, 1)
        query_pos_embed = self.query_pos_embed.view(1, self.image_size[0], self.image_size[1], c).repeat(b, 1, 1, 1)
        #  [B Q t 3]
        query_timesteps, query_positions = (
            query_points[..., :1],
            query_points[..., 1:],
        )
        # grid sample expects coordinates in [-1, 1]
        query_pos_embed_spatial = F.grid_sample(
            query_pos_embed.permute(0, 3, 2, 1),  # we swap h and w due to how grid_sample works
            (query_positions.unsqueeze(1) / torch.tensor(self.image_size, device=query_positions.device)) * 2 - 1,
            align_corners=False,
        ).permute(
            0, 2, 3, 1
        )  # [b Q t c]
        point_query_tokens = tiled_point_query_tokens + query_pos_embed_spatial
        # NOTE: query hints are only implemented in jax but not in pytorch
        # the two masks below are used to not add point query (if queried later in
        # online tracking)
        if t == 1:  # online tracking
            mask_and_query_tokens = torch.where(query_timesteps.unsqueeze(1) == 0, point_query_tokens, mask_tokens)
        else:
            queries_are_late = query_timesteps >= t
            queries_are_early = query_timesteps < 0
            mask_and_query_tokens = mask_tokens.scatter(
                dim=1,
                index=query_timesteps.unsqueeze(1).long().clamp(0, t - 1).repeat(1, 1, 1, c),
                src=point_query_tokens,
            )
            mask_and_query_tokens = torch.where(
                (queries_are_late | queries_are_early).unsqueeze(1),
                mask_tokens,
                mask_and_query_tokens,
            )
        is_unknown_token = torch.arange(t, device=query_points.device)[None, :, None, None] < query_timesteps.unsqueeze(
            1
        )
        mask_and_query_tokens = torch.where(is_unknown_token, unknown_tokens, mask_and_query_tokens)
        return mask_and_query_tokens

    def prediction_heads(self, x):
        soft_argmax_threshold = 20
        softmax_temperature = 0.5
        track_logits = self.coordinate_head(x.float())
        position_y, position_x = track_logits.chunk(2, dim=-1)
        argmax_y, argmax_x = position_y.argmax(dim=-1, keepdim=True), position_x.argmax(dim=-1, keepdim=True)
        index = torch.arange(position_y.shape[-1], device=x.device).repeat(*argmax_y.shape[:-1], 1)
        mask_y = (torch.abs(argmax_y - index) <= soft_argmax_threshold).float()
        mask_x = (torch.abs(argmax_x - index) <= soft_argmax_threshold).float()
        probs_y = F.softmax(position_y * softmax_temperature, dim=-1) * mask_y
        probs_x = F.softmax(position_x * softmax_temperature, dim=-1) * mask_x
        probs_y = probs_y / probs_y.sum(dim=-1, keepdim=True)
        probs_x = probs_x / probs_x.sum(dim=-1, keepdim=True)
        tracks_y = torch.sum(probs_y * index, dim=-1)[..., None]
        tracks_x = torch.sum(probs_x * index, dim=-1)[..., None]
        tracks = torch.cat([tracks_y, tracks_x], axis=-1)
        tracks += 0.5
        visible_logits = self.visible_head(x)
        return tracks, track_logits, visible_logits, mask_y, mask_x

    def forward(self, video, query_points=None, state=None):
        # video.shape
        b, t, _, _, _ = video.shape
        # [b, t, h, w, 3] -> [b, t, 3, h, w]
        video_tokens = self.lin_proj(einops.rearrange(video, "b t h w c -> (b t) c h w"))
        _, _, h, w = video_tokens.shape
        video_tokens = einops.rearrange(video_tokens, "(b t) c h w -> b t (h w) c", b=b, t=t)
        video_tokens = video_tokens + self.image_pos_emb.unsqueeze(0)
        if state is not None:
            # in online tracking, we put query "back in time"
            if query_points is None:
                query_points = state.query_points
            query_points = torch.cat([query_points[..., :1] - state.step, query_points[..., 1:]], dim=-1)
            step = state.step
        else:
            step = 0
        point_tokens = self.embed_queries(t, query_points)  # [b t Q c]
        x = torch.cat([video_tokens, point_tokens], dim=2)  # [b t (h * w + Q) c]
        ssm_cache = []
        use_linear_scan = not self.training
        for blk, cache in zip(self.blocks, state.hidden_state if state is not None else [None] * 12):
            if self.use_checkpointing:
                x, ssm_cache_layer = torch.utils.checkpoint.checkpoint(
                    blk, x, cache, use_linear_scan, use_reentrant=False
                )
            else:
                x, ssm_cache_layer = blk(x, cache=cache, use_linear_scan=use_linear_scan)
            ssm_cache.append(ssm_cache_layer)
        x = self.encoder_norm(x)
        video_tokens, point_tokens = x[:, :, : h * w, :], x[:, :, h * w :, :]
        return (
            *self.prediction_heads(point_tokens),
            TAPNextTrackingState(
                step=step + t,
                query_points=state.query_points if state is not None else query_points,
                hidden_state=ssm_cache,
            ),
        )


torch.export.register_dataclass(TAPNextTrackingState)


def flatten_tracking_state(state, _):
    return (
        state.step,
        state.query_points,
        [[st for st in h] for h in state.hidden_state],
    )


torch.fx._pytree.register_pytree_flatten_spec(  # pylint: disable=protected-access
    TAPNextTrackingState, flatten_tracking_state
)


def restore_model_from_jax_checkpoint(model, ckpt_path):
    """Restores a TAPNext model from a JAX checkpoint."""
    ckpt = {k: v for k, v in np.load(ckpt_path).items()}
    model.lin_proj.weight.data.copy_(torch.tensor(ckpt["backbone/embedding/kernel"][0]).permute(3, 2, 0, 1))
    model.lin_proj.bias.data.copy_(torch.tensor(ckpt["backbone/embedding/bias"]))
    model.mask_token.data.copy_(torch.tensor(ckpt["backbone/mask_token"]))
    model.point_query_token.data.copy_(torch.tensor(ckpt["backbone/point_query_token"]))
    model.unknown_token.data.copy_(torch.tensor(ckpt["backbone/unknown_token"]))
    model.image_pos_emb.data.copy_(torch.tensor(ckpt["backbone/pos_embedding"]))
    model.encoder_norm.weight.data.copy_(torch.tensor(ckpt["backbone/Transformer/encoder_norm/scale"]))
    model.encoder_norm.bias.data.copy_(torch.tensor(ckpt["backbone/Transformer/encoder_norm/bias"]))
    for layer in range(12):
        # convert ssm part
        prefix = f"backbone/Transformer/encoderblock_{layer}/ssm_block"
        ssm_params = {
            key: torch.tensor(ckpt[f"{prefix}/" + re.sub("weight", "kernel", re.sub(r"\.", "/", key))])
            for key, _ in model.blocks[layer].ssm_block.named_parameters()
        }
        for key in ssm_params:
            if "weight" in key:
                ssm_params[key] = ssm_params[key].T
        model.blocks[layer].ssm_block.load_state_dict(ssm_params)

        # convert vit part
        vit_params = {
            re.sub(f"backbone/Transformer/encoderblock_{layer}/vit_block/", "", k): v
            for k, v in ckpt.items()
            if f"backbone/Transformer/encoderblock_{layer}/vit_block" in k
        }
        torch_vit_params = {}
        torch_vit_params["ln_1.weight"] = vit_params["LayerNorm_0/scale"]
        torch_vit_params["ln_1.bias"] = vit_params["LayerNorm_0/bias"]
        torch_vit_params["ln_2.weight"] = vit_params["LayerNorm_1/scale"]
        torch_vit_params["ln_2.bias"] = vit_params["LayerNorm_1/bias"]
        torch_vit_params["mlp.0.weight"] = vit_params["MlpBlock_0/Dense_0/kernel"].T
        torch_vit_params["mlp.0.bias"] = vit_params["MlpBlock_0/Dense_0/bias"]
        torch_vit_params["mlp.3.weight"] = vit_params["MlpBlock_0/Dense_1/kernel"].T
        torch_vit_params["mlp.3.bias"] = vit_params["MlpBlock_0/Dense_1/bias"]
        torch_vit_params["self_attention.in_proj_weight"] = np.concatenate(
            [
                vit_params["MultiHeadDotProductAttention_0/query/kernel"].reshape(768, 768).T,
                vit_params["MultiHeadDotProductAttention_0/key/kernel"].reshape(768, 768).T,
                vit_params["MultiHeadDotProductAttention_0/value/kernel"].reshape(768, 768).T,
            ],
            axis=0,
        )
        torch_vit_params["self_attention.in_proj_bias"] = np.concatenate(
            [
                vit_params["MultiHeadDotProductAttention_0/query/bias"].flatten(),
                vit_params["MultiHeadDotProductAttention_0/key/bias"].flatten(),
                vit_params["MultiHeadDotProductAttention_0/value/bias"].flatten(),
            ]
        )
        torch_vit_params["self_attention.out_proj.weight"] = (
            vit_params["MultiHeadDotProductAttention_0/out/kernel"].reshape(768, 768).T
        )
        torch_vit_params["self_attention.out_proj.bias"] = vit_params[
            "MultiHeadDotProductAttention_0/out/bias"
        ].flatten()
        for k in torch_vit_params:
            torch_vit_params[k] = torch.tensor(np.array(torch_vit_params[k]))
        model.blocks[layer].vit_block.load_state_dict(torch_vit_params)
    model.visible_head[0].weight.data.copy_(torch.from_numpy(ckpt["visible_head/layers_0/kernel"].T))
    model.visible_head[0].bias.data.copy_(torch.from_numpy(ckpt["visible_head/layers_0/bias"]))
    model.visible_head[1].weight.data.copy_(torch.from_numpy(ckpt["visible_head/layers_1/scale"]))
    model.visible_head[1].bias.data.copy_(torch.from_numpy(ckpt["visible_head/layers_1/bias"]))
    model.visible_head[3].weight.data.copy_(torch.from_numpy(ckpt["visible_head/layers_3/kernel"].T))
    model.visible_head[3].bias.data.copy_(torch.from_numpy(ckpt["visible_head/layers_3/bias"]))
    model.visible_head[4].weight.data.copy_(torch.from_numpy(ckpt["visible_head/layers_4/scale"]))
    model.visible_head[4].bias.data.copy_(torch.from_numpy(ckpt["visible_head/layers_4/bias"]))
    model.visible_head[6].weight.data.copy_(torch.from_numpy(ckpt["visible_head/layers_6/kernel"].T))
    model.visible_head[6].bias.data.copy_(torch.from_numpy(ckpt["visible_head/layers_6/bias"]))

    model.coordinate_head[0].weight.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_0/kernel"].T))
    model.coordinate_head[0].bias.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_0/bias"]))
    model.coordinate_head[1].weight.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_1/scale"]))
    model.coordinate_head[1].bias.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_1/bias"]))
    model.coordinate_head[3].weight.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_3/kernel"].T))
    model.coordinate_head[3].bias.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_3/bias"]))
    model.coordinate_head[4].weight.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_4/scale"]))
    model.coordinate_head[4].bias.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_4/bias"]))
    model.coordinate_head[6].weight.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_6/kernel"].T))
    model.coordinate_head[6].bias.data.copy_(torch.from_numpy(ckpt["coordinate_head/layers_6/bias"]))
    return model


# -- Utilities --


__stop = False


def should_stop():
    return __stop


def register_sigusr1_handler():
    def signal_handler_usr1(sig, frame):
        logging.getLogger(__name__).info(f"Got SigUSR1. Stopping...", flush=True)
        global __stop
        __stop = True

    logging.getLogger(__name__).info("Registering handler for SigUSR1...")
    signal.signal(signal.SIGUSR1, signal_handler_usr1)


def process_shards(
    input_dir: Path,
    is_processed_fn: Callable[[Path], bool],
    process_fn: Callable[[Path], None],
    fs_sync_dir: Path | None = None,
    fs_sync_delay: float = 10,
    input_glob: str = "*.tar",
    stop_on_sigusr1: bool = True,
    rank: int | None = None,
    use_tqdm: bool = True,
):
    if rank is None:
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
        rank = int(os.environ["CUDA_VISIBLE_DEVICES"])
    hostname = socket.gethostname()
    prefix = f"{hostname}-{rank}"

    if stop_on_sigusr1:
        register_sigusr1_handler()

    input_files = list(input_dir.rglob(input_glob))
    logging.getLogger(__name__).info(f"[{prefix}] Found {len(input_files)} files to process.")
    random.Random(hash(prefix) + os.getpid()).shuffle(input_files)
    fs_sync_dir = fs_sync_dir or input_dir
    fs_sync_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(input_files, disable=(not use_tqdm)):
        if stop_on_sigusr1 and should_stop():
            break

        if is_processed_fn(f):
            continue

        # We use these weird lockfiles to coordinate across ranks on slightly unstable filesystems
        # There, classic mutex lock files often don't work well and still lead to race conditions and deadlocks
        lockfiles = list((fs_sync_dir / f.relative_to(input_dir)).parent.glob(f"{f.stem}.*.lock"))
        if len(lockfiles) > 0:
            continue
        lockfile = fs_sync_dir / f.relative_to(input_dir).parent / f"{f.stem}.{prefix}.lock"
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        with lockfile.open("w") as f_:
            f_.write("hello")
        time.sleep(fs_sync_delay)
        lockfiles = list((fs_sync_dir / f.relative_to(input_dir)).parent.glob(f"{f.stem}.*.lock"))
        if len(lockfiles) > 1:
            lockfiles_sorted = sorted(lockfiles, key=lambda f: f.stat().st_mtime)
            lockfiles_first = [
                lf for lf in lockfiles_sorted if lf.stat().st_mtime == lockfiles_sorted[0].stat().st_mtime
            ]
            if len(lockfiles_first) > 1:
                lockfile_first = sorted(lockfiles_first, key=lambda f: f.name)[0]
            else:
                lockfile_first = lockfiles_first[0]
            if lockfile != lockfile_first:
                lockfile.unlink()
                continue

        logging.getLogger(__name__).info(f"[{prefix}] Processing {f}...")
        try:
            process_fn(f)
            assert is_processed_fn(f)
            logging.getLogger(__name__).info(f"[{prefix}] Done processing {f}.")
        except Exception as e:
            logging.exception(f"[{prefix}] Error processing {f}.")
        finally:
            lockfile.unlink()

    print("processed all shards on rank", rank)


class Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def get_rank_world():
    # Prefer generic envs if you set them, else fall back to SLURM.
    world = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)
    rank = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)
    if world < 1:  # safety
        world = 1
        rank = 0
    hostname = socket.gethostname()
    return rank, world, hostname


# -- TapNext Processing --


def init_tapnext():
    # this is the training resolution AFAIK
    # https://github.com/google-deepmind/tapnet?tab=readme-ov-file#checkpoints
    model = TAPNext(image_size=(256, 256))

    # find ckpt path that exists on this system

    ckpt_path = None
    for path in possible_ckpt_path:
        try:
            if Path(path).exists():
                ckpt_path = path
                break
        except:
            continue

    if ckpt_path is None:
        raise ValueError(
            f"Could not find any checkpoint on this system from {possible_ckpt_path}. Download from https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz"
        )

    model = restore_model_from_jax_checkpoint(model, ckpt_path)
    model.to(device=DEVICE)  # , dtype=DTYPE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if COMPILE:
        model.forward = torch.compile(model.forward)

    return model


def fuse_tracks(pred_fwd, pred_bwd, start_times):
    """
    pred_fwd, pred_bwd: (B, T, Q, C)
    start_times: (B, Q, 1) or (1, Q, 1), Q: number of queries
    returns fused: (B, T, Q, C)
    """
    B, T, Q, C = pred_fwd.shape
    t_idx = torch.arange(T, device=pred_fwd.device).view(1, T, 1, 1)
    if start_times.shape[0] == 1 and B != 1:
        start_times = start_times.expand(B, -1, -1)
    elif start_times.shape[0] != B:
        raise ValueError(f"start_times batch size must be 1 or {B}, got {start_times.shape[0]}")
    return torch.where(t_idx <= start_times.view(B, 1, Q, 1), pred_bwd, pred_fwd)


def get_window(coord, softmax, radius: int = 8):
    b = coord.shape[0]
    start = torch.floor(coord - radius - 0.5).int()
    start.clamp_(min=0)
    indices = start + torch.arange(radius * 2 + 1, device=softmax.device).repeat(b, 1)
    # this is to simulate one corner case of jax implementation
    shift = (indices.max(1).values - softmax.shape[1] + 1).clamp(min=0)
    indices -= shift.unsqueeze(1)
    softmax = softmax.gather(dim=1, index=indices)
    return softmax, indices + 0.5


def tracker_certainty(coord_yx, track_logits, radius=8):
    track_logits = track_logits.float()
    # Computes the certainty of the tracker.
    shape = coord_yx.shape[:-1]
    coord_yx = coord_yx.flatten(0, -2)
    track_logits = track_logits.flatten(0, -2)
    # track_logits.shape == [b, 512]
    # coord_yx.shape == [b, 2]
    logits_y, logits_x = track_logits.chunk(2, dim=-1)
    track_softmax_y = F.softmax(logits_y, dim=-1)
    track_softmax_x = F.softmax(logits_x, dim=-1)
    sm_y, coord_y = get_window(coord_yx[:, 0:1], track_softmax_y)
    sm_x, coord_x = get_window(coord_yx[:, 1:2], track_softmax_x)
    sm = sm_y[..., :, None] * sm_x[..., None, :]
    grid_y = coord_y[:, :, None].expand(-1, -1, coord_x.shape[1])
    grid_x = coord_x[:, None, :].expand(-1, coord_y.shape[1], -1)
    grid = torch.stack([grid_y, grid_x], dim=-1)
    in_radius = ((grid - coord_yx[:, None, None]) ** 2).sum(-1) <= ((radius**2) + 1e-8)
    return (sm * in_radius).sum(-1).sum(-1).reshape(*shape, 1)


def certainty_chunked(coord_yx, track_logits, chunk_q=128):
    B, T, Q, _ = coord_yx.shape
    outs = []
    for q0 in range(0, Q, chunk_q):
        q1 = min(Q, q0 + chunk_q)
        outs.append(tracker_certainty(coord_yx[:, :, q0:q1], track_logits[:, :, q0:q1]))
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=2)


def run_tapnext(
    model: TAPNext,
    video: Float[torch.Tensor, "b c t h w"],
    num_tracks: int,
):
    # video in [-1, 1]
    video = TVT.functional.resize(
        video, TAPNEXT_INPUT_SIZE, interpolation=TVT.InterpolationMode.BICUBIC, antialias=True
    )

    video = video.to(dtype=DTYPE, device=DEVICE)

    grid = torch.rand((1, num_tracks, 2), dtype=DTYPE, device=DEVICE)
    grid = grid.mul(torch.tensor(TAPNEXT_INPUT_SIZE, dtype=DTYPE, device=DEVICE)[None, None, :].sub(1))
    grid = grid.add(0.5)  # necessary because tapnext samples with align_corners=False

    T = video.shape[2]
    queries_per_frame = (num_tracks + T - 1) // T
    start_times = torch.arange(T, device=DEVICE).repeat_interleave(queries_per_frame)
    start_times = start_times[:num_tracks].view(1, num_tracks, 1).to(dtype=DTYPE)
    query_points = torch.cat([start_times, grid], dim=-1).repeat(video.shape[0], 1, 1)
    start_times = start_times.repeat(video.shape[0], 1, 1)

    # channels last!!
    video = video.movedim(1, -1)

    def _run_once(video_seq, query_points):
        with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=True):
            with torch.no_grad():
                pred_tracks, pred_tracks_logits, pred_visible_logits, _, _, _ = model(
                    video=video_seq, query_points=query_points
                )

        return pred_tracks, pred_tracks_logits, pred_visible_logits

    pred_tracks_fwd, pred_tracks_logits_fwd, pred_visible_logits_fwd = _run_once(video, query_points)

    video_rev = torch.flip(video, dims=[1])
    query_points_rev = query_points.clone()
    query_points_rev[:, :, 0] = video.shape[1] - 1 - query_points_rev[:, :, 0]
    pred_tracks_bwd, pred_tracks_logits_bwd, pred_visible_logits_bwd = _run_once(video_rev, query_points_rev)
    pred_tracks_bwd = torch.flip(pred_tracks_bwd, dims=[1])
    pred_tracks_logits_bwd = torch.flip(pred_tracks_logits_bwd, dims=[1])
    pred_visible_logits_bwd = torch.flip(pred_visible_logits_bwd, dims=[1])

    pred_tracks = fuse_tracks(pred_tracks_fwd, pred_tracks_bwd, start_times)
    pred_tracks_logits = fuse_tracks(pred_tracks_logits_fwd, pred_tracks_logits_bwd, start_times)
    pred_visible_logits = fuse_tracks(pred_visible_logits_fwd, pred_visible_logits_bwd, start_times)

    del pred_tracks_fwd, pred_tracks_bwd, pred_tracks_logits_fwd, pred_tracks_logits_bwd
    torch.cuda.empty_cache()

    # Compute visibility and certainty
    visibility_logits = pred_visible_logits[:, :, :, 0]

    # Index of the frame a track was initially queried at.
    # Might be useful to de-bias training
    query_frame = query_points[:, :, 0]

    out_dict = {
        "logits_visible.npy": visibility_logits.half().cpu().numpy(),
        "query_frame_index.npy": query_frame.byte().cpu().numpy(),
    }

    # We rescale tracks such that [-1, 1] specifies the *edges* of the leftmost and rightmost pixels.
    tracks_normalized = pred_tracks.div(pred_tracks.new_tensor(TAPNEXT_INPUT_SIZE)).mul(2).sub(1)
    certainty = certainty_chunked(pred_tracks, pred_tracks_logits)[:, :, :, 0]
    out_dict.update(
        {
            "tracks_yx.npy": tracks_normalized.half().cpu().numpy(),
            "certainty.npy": certainty.half().cpu().numpy(),
        }
    )

    return out_dict


# --- Data loading ---


def wds_filter(sample: dict | None) -> bool:
    # in case of failure `sample is None` -> returns `False` and is removed.
    return sample is not None


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs) -> dict[str, Any]:
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}  # remove keys with "__"
    for s in samples:
        [batched[key].append(s[key]) for key in batched]
    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.tensor(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.tensor(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])
    return result


def out_name_for_shard(input_shard: Path) -> str:
    if input_shard.suffix == ".tar":
        return input_shard.with_suffix("").name + SUFFIX  # data-00000 + <suffix>.tar
    return input_shard.name + SUFFIX


# --- Video processing ---


def transcode_h264(
    frames: torch.Tensor,
    fps: int | None = None,
    crf: int = 20,
    preset: str = "medium",
    input_args: list = ["pipe:"],
    input_kwargs: dict | None = None,
    output_args: list = ["pipe:"],
    output_kwargs: dict | None = None,
) -> bytes | None:
    """
    frames: (T,H,W,3). Prefer uint8 [0,255]. Floats in [0,1] also supported.
    Returns MP4 (H.264) bytes or None on error.
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"expected (T,H,W,3), got {frames.shape}"
    t, h, w, _ = frames.shape
    if t == 0:
        return b""

    # Build sane defaults
    if input_kwargs is None:
        input_kwargs = {
            "format": "rawvideo",
            "pix_fmt": "rgb24",
        }
    # TODO: MPEG decoding instead of mp4 decoding
    if output_kwargs is None:
        output_kwargs = {
            "format": "mp4",
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "crf": crf,
            "preset": preset,
            "movflags": "+frag_keyframe+empty_moov",  # for web playback or "+frag_keyframe+empty_moov" for streamable mp4
        }

    # Convert to uint8 without breaking ranges
    x = frames.cpu().numpy()

    if x.dtype != np.uint8:
        x = (x.clip(0, 1) * 255.0).round().astype(np.uint8)

    x = np.ascontiguousarray(x)

    # FPS handling
    ikw = dict(input_kwargs)
    okw = dict(output_kwargs)
    if fps is not None:
        if not (isinstance(fps, int) and fps > 0):
            raise ValueError(f"fps must be a positive int, got {fps}")
        ikw["r"] = fps  # interpret input at fps
        okw["r"] = fps  # tag output at same fps
    # else: ffmpeg will assume 25 fps for rawvideo input

    # Build and run ffmpeg
    process = (
        ffmpeg.input(*input_args, **ikw, s=f"{w}x{h}")
        .output(*output_args, **okw, an=None)  # no audio
        .overwrite_output()
        .global_args("-loglevel", "error")  # quieter stderr
    )

    proc = process.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    stdout_data, stderr_data = proc.communicate(input=x.tobytes())

    if proc.returncode != 0:
        # stderr already minimal due to -loglevel error
        err = stderr_data.decode("utf-8", errors="ignore")
        print(f"ffmpeg failed (code {proc.returncode}): {err}", flush=True)
        return None

    # Collect bytes
    with io.BytesIO() as buf:
        buf.write(stdout_data)
        proc.wait()
        buf.seek(0)
        return buf.getvalue()


def transcode_h264_wrapper(args):
    frames, fps = args
    return transcode_h264(frames, fps)


def augment(
    clip,
    interpolation=TVT.InterpolationMode.BICUBIC,
    size=(256, 256),
) -> torch.Tensor:
    if size is not None and size[0] is not None:
        clip = TVT.functional.resize(clip, size, interpolation=interpolation, antialias=True)

        # Ensure both H and W are divisible by 2 WITHOUT another resample
        H, W = clip.shape[-2:]
        clip = clip[..., : H - (H % 2), : W - (W % 2)]

    # normalize
    clip = (clip - 0.5) / 0.5
    clip = clip.clamp(-1.0, 1.0)
    return einops.rearrange(clip, "t c h w -> c t h w")


def load_mp4(
    data: bytes,
    augment_args={},
    min_fps=None,
):
    data_dict = {}
    status = -1

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        try:
            tmp.write(data)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                return {}, -1

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or np.isnan(fps) or fps < 1e-3 or (min_fps is not None and fps < min_fps):
                cap.release()
                del cap
                gc.collect()
                return {}, -1

            frames = []
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if len(frames) == 0:
                return {}, -1

            # TODO: Issue: for very long or high-fps videos this can easily exceed memory limits.
            seq = np.stack(frames, axis=0)  # (T,H,W,3)
            seq = torch.from_numpy(seq)  # uint8 [0,255]
            seq = einops.rearrange(seq, "t h w c -> t c h w")
            seq = seq.float() / 255.0  # [0,1]

            seq = augment(seq, **augment_args)
            data_dict["x"] = seq
            data_dict["fps"] = float(fps)
            status = 1
        except Exception as e:
            print(f"Caught {e}.", flush=True)
            data_dict = {}
            status = -1
        finally:
            pass
        return data_dict, status


USE_ALARM = hasattr(signal, "SIGALRM")


def safe_load_mp4(data, **kwargs):
    if USE_ALARM:
        with time_limit(60):  # kill after 1 min
            return load_mp4(data, **kwargs)
    else:
        return load_mp4(data, **kwargs)


def decode_mp4(
    sample: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    return_dict = {}
    found_valid_ext = False
    assert "__key__" in sample.keys(), f"{sample.keys()}"
    key = sample["__key__"]
    for k, v in sample.items():
        if k.endswith(MP4_SUFFIX):
            found_valid_ext = True
            data, status = safe_load_mp4(v, **kwargs)
            if status <= 0:
                print(f"[DEBUG] failed to decode mp4 {key=}, {k=}, {status=}", flush=True)
                return None
            else:
                for k2, v2 in data.items():
                    return_dict[k2] = v2
        else:
            return_dict[k] = v
    if not found_valid_ext:
        print(f"couldn't find valid video extension ({MP4_SUFFIX}) in {key=}", flush=True)
        return None
    return return_dict


# --- Processing Utilities ---


def split_video(video: torch.Tensor, max_seq_len: int, min_seq_len: int) -> list[torch.Tensor]:
    T = video.shape[1]
    if T < min_seq_len:
        return []
    if T <= max_seq_len:
        return [video]

    splits = []
    n_full = T // max_seq_len
    for i in range(n_full):
        start = i * max_seq_len
        end = start + max_seq_len
        splits.append(video[:, start:end])

    return splits


def make_clip_key(key: Any, clip_index: int, num_clips: int) -> str:
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    key = str(key)
    if num_clips <= 1:
        return key
    return f"{key}_clip{clip_index:03d}"


# --- Main Loop ---


@click.command()
@click.option("--input_dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output_dir", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("--sync_dir", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("--fps", type=float, default=12)
@click.option("--min_seq_len", type=int, default=16)
@click.option("--max_seq_len", type=int, default=128)  # TAPNEXT deteriorates after 150 frames
@click.option("--nr_workers", type=int, default=8)
@click.option("--no_tqdm", is_flag=True, default=False, help="Disable tqdm progress bar.")
@click.option(
    "--num-tracks",
    type=int,
    default=1024,
    help="Fixed number of random queries per video.",
)
@click.option(
    "--out-size",
    type=int,
    default=480,
    help="Output size of the video frames in pixels.",
)
def process(
    input_dir: Path,
    output_dir: Path,
    sync_dir: str,
    min_tracking_fps: float = 10,
    fps: float | None = 12,
    min_seq_len: int = 16,
    max_seq_len: int = 128,
    nr_workers: int = 8,
    no_tqdm: bool = False,
    video_key: str = "video.mp4",
    num_tracks: int = 1024,
    out_size: int = 480,
):
    input_base_path = Path(input_dir)
    output_base_path = Path(output_dir)
    sync_base_path = Path(sync_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)
    sync_base_path.mkdir(parents=True, exist_ok=True)

    model = init_tapnext()

    def process_shard(input_shard: Path):
        dataset = wds.DataPipeline(
            wds.shardlists.SimpleShardList(str(input_shard)),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.split_by_worker,
            wds.decode(),
            wds.map(
                partial(
                    decode_mp4,
                    min_fps=min_tracking_fps,
                    augment_args={
                        "size": (out_size,),  # shorter edge
                    },
                )
            ),
            wds.select(wds_filter),
            wds.batched(1, collation_fn=dict_collation_fn),
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=nr_workers,
        )

        ctx = get_context("spawn")
        MAX_TASKS = 256

        def new_pool():
            return ctx.Pool(processes=1, maxtasksperchild=MAX_TASKS)

        transcoding_pool = new_pool()
        tasks_since_recreate = 0

        out_shard_path = output_base_path / out_name_for_shard(input_shard)

        with wds.TarWriter(str(out_shard_path)) as sink:
            for batch in tqdm(loader, desc=f"Preprocessing {input_shard}", disable=no_tqdm, miniters=100):
                batch: dict[str, Any]
                # format `b c t h w` in [-1 , 1]
                video: torch.Tensor = batch["x"][0].float()

                video_fps = float(batch["fps"][0]) if "fps" in batch else 30.0
                if fps is not None:
                    stride = max(int(round(video_fps / fps)), 1)
                    video = video[:, ::stride]
                    video_fps = video_fps / stride

                clips = split_video(video, max_seq_len, min_seq_len)
                if len(clips) == 0:
                    continue

                source_key = batch["__key__"][0]
                for clip_index, clip in enumerate(clips):
                    transcoding_future = transcoding_pool.apply_async(
                        transcode_h264_wrapper,
                        (
                            (
                                einops.rearrange(clip, "c t h w -> t h w c") * 0.5 + 0.5,
                                int(round(video_fps)),
                            ),
                        ),
                    )

                    tapnext_outputs = run_tapnext(
                        model,
                        clip.unsqueeze(0),
                        num_tracks=num_tracks,
                    )

                    buffer = transcoding_future.get()
                    tasks_since_recreate += 1
                    if tasks_since_recreate >= MAX_TASKS:
                        transcoding_pool.close()
                        transcoding_pool.join()
                        transcoding_pool = new_pool()
                        tasks_since_recreate = 0

                    if buffer is None:
                        del tapnext_outputs, transcoding_future
                        continue
                    try:
                        data = {k: v[0] for k, v in tapnext_outputs.items()}
                        data["__key__"] = make_clip_key(source_key, clip_index, len(clips))
                        data[video_key] = buffer

                        for k, v in batch.items():
                            if k in ["x", "fps", "__url__", "meta.json", "__key__"]:
                                continue
                            if isinstance(v, torch.Tensor):
                                data[k] = v[0]
                            elif isinstance(v, np.ndarray):
                                data[k] = v[0]
                            elif isinstance(v, (list, tuple)):
                                data[k] = v[0]
                            else:
                                data[k] = v

                        sink.write(data)

                    except Exception as e:
                        print(f"Caught {e=}", flush=True)
                    finally:
                        del tapnext_outputs, transcoding_future
                del video, batch, clips
                gc.collect()
        transcoding_pool.close()
        transcoding_pool.join()
        gc.collect()

    process_shards(
        input_dir=input_base_path,
        is_processed_fn=lambda f: (output_base_path / out_name_for_shard(f)).exists(),
        fs_sync_dir=sync_base_path,
        process_fn=process_shard,
        stop_on_sigusr1=True,
        use_tqdm=True,
        fs_sync_delay=1,
        rank=get_rank_world()[0],
    )

    print("Exited gracefully.")
    exit(0)


if __name__ == "__main__":
    process()
