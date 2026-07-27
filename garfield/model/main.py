import torch

from torch import nn
from .encoder import TrajDistribEncoder


class SparsityScheduler(nn.Module):

    def __init__(
        self,
        start_sparsity: float,
        end_sparsity: float,
        total_steps: int,
    ):
        super().__init__()
        assert total_steps > 0
        self.register_buffer("s0", torch.tensor(start_sparsity, dtype=torch.float32))
        self.register_buffer("s1", torch.tensor(end_sparsity, dtype=torch.float32))
        self.register_buffer("total_steps", torch.tensor(total_steps, dtype=torch.float32))
        self.scheduler_step = nn.Parameter(torch.tensor(0), requires_grad=False)

    @torch.no_grad()
    def forward(self):
        progress = torch.clamp(self.scheduler_step.float() / self.total_steps, 0.0, 1.0)
        sparsity = self.s0 + progress * (self.s1 - self.s0)
        if self.training:
            self.scheduler_step += 1
        return sparsity

class GARFIELD(nn.Module):
    def __init__(
        self,
        encoder: TrajDistribEncoder,
        decoder: nn.Module,
        sparsity_scheduler: SparsityScheduler,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sparsity_scheduler = sparsity_scheduler
        self.freeze_encoder = freeze_encoder

        self.encoder.requires_grad_(not freeze_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def make_is_query(
        self,
        tracks: torch.Tensor,
        cond_frame_idx: torch.Tensor,
    ):
        prob_is_query = self.sparsity_scheduler().item()
        B, T, N, _ = tracks.shape
        device = tracks.device
        B_idx = torch.arange(tracks.shape[0], device=device)
        is_query = torch.rand((B, T, N), device=device) < prob_is_query
        is_query[B_idx, cond_frame_idx] = False
        return is_query
        
    def forward(
        self,
        cond_frame: torch.Tensor,
        cond_frame_idx: torch.Tensor,
        tracks: torch.Tensor,
        **kwargs,
    ):
        is_query = self.make_is_query(
            tracks=tracks,
            cond_frame_idx=cond_frame_idx,
        )
        enc_precomputed = self.encoder.precompute(
            cond_frame=cond_frame,
            tracks=tracks,
        )
        latent = self.encoder(
            tracks=tracks,
            is_query=is_query,
            precomputed_kwargs=enc_precomputed,
            **kwargs
        )
        return self.decoder.loss(
            x=tracks,
            latent=latent,
            cond_frame_idx=cond_frame_idx,
            tracks_for_encoder=tracks,
            **kwargs,
        )
