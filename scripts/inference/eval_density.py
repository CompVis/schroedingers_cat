import math
import random

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from einops import rearrange, repeat
from pathlib import Path
from time import perf_counter
from typing import Annotated, TypeVar

import numpy as np
import torch
from tqdm import tqdm

from garfield.data import TrackerVideoDataModule
from garfield.model.density_decoder import DensityEstimator, make_density_decoder
from garfield.model.encoder import TrajDistribEncoder, make_encoder


torch.set_float32_matmul_precision("high")

Component = TypeVar("Component", bound=torch.nn.Module)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True, help="Path to the encoder checkpoint.")
    parser.add_argument("--density", type=str, required=True, help="Path to the density decoder checkpoint.")
    parser.add_argument("--tarbase", type=str, required=True, help="Directory containing the evaluation tar shards.")
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Shard glob relative to --tarbase. By default all tar files below --tarbase are used.",
    )
    parser.add_argument("--num-samples", type=int, required=True, help="Number of dataset samples to evaluate.")
    parser.add_argument(
        "--sparsity",
        type=float,
        required=True,
        help="Probability that a point is hidden/query. Timestep zero is always conditional.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--horizon", type=int, default=31, help="Number of trajectory timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--compile", action="store_true", help="Compile encoder and decoder forward methods.")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA.")
    return parser


def load_component(
    ckpt_path: str,
    make_component_fn: Callable[[], Component],
    component_name: str,
) -> Component:
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        component = make_component_fn()
        component.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load {component_name} from checkpoint '{ckpt_path}'. "
            f"Check that it contains weights compatible with {make_component_fn.__name__}. "
            f"Original error: {error}"
        ) from error
    component.eval()
    return component


def load_model(
    device: torch.device,
    encoder_checkpoint: str,
    density_checkpoint: str,
    compile_model: bool,
) -> tuple[TrajDistribEncoder, DensityEstimator]:
    encoder = load_component(encoder_checkpoint, make_encoder, "encoder").to(device=device, dtype=torch.float32)
    decoder = load_component(density_checkpoint, make_density_decoder, "density decoder").to(
        device=device,
        dtype=torch.float32,
    )
    compile_kwargs = {
        "mode": "reduce-overhead",
        "fullgraph": True,
        "disable": not compile_model,
    }
    encoder.precompute = torch.compile(encoder.precompute, **compile_kwargs)
    encoder.forward = torch.compile(encoder.forward, **compile_kwargs)
    decoder.decode_latent = torch.compile(decoder.decode_latent, **compile_kwargs)
    return encoder, decoder


def make_dataloader(
    tarbase: str,
    shards: str | None,
    batch_size: int,
    num_workers: int,
    image_size: int,
    horizon: int,
    num_tracks: int,
) -> Iterable[dict[str, object]]:
    data_config = {
        "tar_base": tarbase,
        "shards": shards,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": 0,
        "simple_heuristic": True,
        "filter_static_camera": False,
        "decode_kwargs": {
            "track_key": "tracks_yx",
            "image_size": image_size,
            "num_steps": horizon + 1,
            "num_tracks": num_tracks,
            "frame_skip": 1,
            "center_crop": True,
            "certainty_threshold": 0.6,
            "return_full_sequence": False,
            "return_video_without_cutting": False,
            "static_camera_flow_mag_threshold": 0.0002,
            "static_camera_fraction_threshold": 0.45,
        },
    }
    data_module = TrackerVideoDataModule(train=data_config, validation=data_config)
    return data_module.val_dataloader()


def make_query_mask(
    batch_size: int,
    horizon: int,
    num_tracks: int,
    sparsity: float,
    device: torch.device,
) -> Annotated[torch.Tensor, "B T N, bool"]:
    is_query = torch.rand((batch_size, horizon, num_tracks), device=device) < sparsity
    is_query[:, 0] = False
    return is_query


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_square_grid_and_oob(num_tokens: int) -> tuple[int, int, bool]:
    side = math.isqrt(num_tokens)
    if side * side == num_tokens:
        return side, side, False
    side = math.isqrt(num_tokens - 1)
    if side * side == num_tokens - 1:
        return side, side, True
    raise ValueError(f"Expected a square heatmap with an optional OOB token, got {num_tokens} tokens.")


def discrete_energy_score(
    ground_truth: Annotated[torch.Tensor, "B T N 2, float"],
    heatmaps: Annotated[torch.Tensor, "B T N L, float"],
    chunk_size: int = 512,
    epsilon: float = 1e-12,
) -> Annotated[torch.Tensor, "B T N, float"]:
    batch_size, horizon, num_tracks, _ = ground_truth.shape
    num_tokens = heatmaps.shape[-1]
    height, width, has_oob = infer_square_grid_and_oob(num_tokens)
    num_grid_tokens = height * width
    dtype = ground_truth.dtype if ground_truth.dtype.is_floating_point else torch.float32
    device = ground_truth.device

    ground_truth = ground_truth.to(device=device, dtype=dtype)
    heatmaps = heatmaps.to(device=device, dtype=dtype)
    ground_truth_in_bounds = ((ground_truth >= 0.0) & (ground_truth <= 1.0)).all(dim=-1)

    x_centers = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y_centers = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
    support = torch.stack([grid_x, grid_y], dim=-1).reshape(num_grid_tokens, 2)

    num_predictions = batch_size * horizon * num_tracks
    targets = ground_truth.reshape(num_predictions, 2)
    if has_oob:
        probabilities = heatmaps[..., :num_grid_tokens].reshape(num_predictions, num_grid_tokens)
    else:
        probabilities = heatmaps.reshape(num_predictions, num_grid_tokens)
    probabilities = probabilities.clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(epsilon)

    distance_to_target = torch.cdist(targets, support)
    target_term = (probabilities * distance_to_target).sum(dim=-1)

    probabilities_transposed = probabilities.mT.contiguous()
    pairwise_term = torch.zeros(num_predictions, device=device, dtype=dtype)
    for start in range(0, num_grid_tokens, chunk_size):
        end = min(start + chunk_size, num_grid_tokens)
        pairwise_distance = torch.cdist(support[start:end], support)
        expected_distance = pairwise_distance @ probabilities_transposed
        probabilities_chunk = probabilities[:, start:end].mT
        pairwise_term += (probabilities_chunk * expected_distance).sum(dim=0)

    energy_score = (target_term - 0.5 * pairwise_term).reshape(batch_size, horizon, num_tracks)
    nan = torch.tensor(float("nan"), device=device, dtype=dtype)
    return torch.where(ground_truth_in_bounds, energy_score, nan)


@torch.inference_mode()
def evaluate_batch(
    batch: dict[str, object],
    encoder: TrajDistribEncoder,
    decoder: DensityEstimator,
    sparsity: float,
    device: torch.device,
    max_batch_size: int,
) -> tuple[Annotated[torch.Tensor, "B, float"], float, float, float]:
    tracks = batch["tracks"]
    cond_frame = batch["cond_frame"]
    is_camera_static = batch["is_camera_static"]
    if not isinstance(tracks, torch.Tensor):
        raise TypeError(f"Expected tensor tracks, got {type(tracks).__name__}.")
    if not isinstance(cond_frame, torch.Tensor):
        raise TypeError(f"Expected tensor cond_frame, got {type(cond_frame).__name__}.")
    if not isinstance(is_camera_static, torch.Tensor):
        raise TypeError(f"Expected tensor is_camera_static, got {type(is_camera_static).__name__}.")

    tracks = tracks[:max_batch_size].to(device=device, dtype=torch.float32)
    cond_frame = cond_frame[:max_batch_size].to(device=device, dtype=torch.float32)
    is_camera_static = is_camera_static[:max_batch_size].to(device=device)
    if cond_frame.ndim == 5:
        cond_frame = cond_frame[:, 0]

    batch_size, horizon, num_tracks, _ = tracks.shape
    is_query = make_query_mask(batch_size, horizon, num_tracks, sparsity=sparsity, device=device)
    masked_tracks = torch.rand_like(tracks)
    masked_tracks[~is_query] = tracks[~is_query]

    synchronize_device(device)
    total_start = perf_counter()
    encoder_start = perf_counter()
    precomputed = encoder.precompute(cond_frame, tracks=masked_tracks)
    precomputed = {name: value.clone() for name, value in precomputed.items()}
    latent = encoder(
        masked_tracks,
        is_query,
        is_camera_static,
        precomputed_kwargs=precomputed,
    ).clone()
    synchronize_device(device)
    encoder_runtime = perf_counter() - encoder_start

    timestep = repeat(
        torch.arange(horizon, device=device, dtype=torch.float32),
        "T -> B T N",
        B=batch_size,
        N=num_tracks,
    )
    timestep = timestep / horizon
    initial_x = repeat(tracks[:, 0, :, 0], "B N -> B T N", T=horizon)
    initial_y = repeat(tracks[:, 0, :, 1], "B N -> B T N", T=horizon)
    synchronize_device(device)
    decoder_start = perf_counter()
    density, out_of_bounds = decoder.decode_latent(
        latent=latent,
        with_oob_tok=True,
        split_bs=1024,
        t_idx=timestep,
        x_idx=initial_x,
        y_idx=initial_y,
    )
    synchronize_device(device)
    decoder_runtime = perf_counter() - decoder_start
    total_runtime = perf_counter() - total_start

    density_tokens = rearrange(density, "B T N H W 1 -> B T N (H W)")
    out_of_bounds = rearrange(out_of_bounds, "B (T N) 1 -> B T N 1", T=horizon, N=num_tracks)
    heatmaps = torch.cat([density_tokens, out_of_bounds], dim=-1)

    energy_score = discrete_energy_score(
        ground_truth=tracks[..., [1, 0]],
        heatmaps=heatmaps,
    )
    sample_scores = torch.nanmean(energy_score.flatten(start_dim=1), dim=1)
    return (
        sample_scores,
        total_runtime / batch_size,
        encoder_runtime / batch_size,
        decoder_runtime / batch_size,
    )


def validate_args(args: Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {args.num_samples}.")
    if not 0.0 <= args.sparsity <= 1.0:
        raise ValueError(f"--sparsity must be in [0, 1], got {args.sparsity}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers cannot be negative, got {args.num_workers}.")
    if args.horizon <= 0:
        raise ValueError(f"--horizon must be positive, got {args.horizon}.")
    if not Path(args.tarbase).is_dir():
        raise ValueError(f"--tarbase must be an existing directory, got '{args.tarbase}'.")


def main(args: Namespace) -> None:
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    encoder, decoder = load_model(
        device=device,
        encoder_checkpoint=args.encoder,
        density_checkpoint=args.density,
        compile_model=args.compile,
    )
    dataloader = make_dataloader(
        tarbase=args.tarbase,
        shards=args.shards,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=encoder.image_encoder.resize,
        horizon=args.horizon,
        num_tracks=encoder.track_identity_table.nr_tracks,
    )

    sample_scores: list[torch.Tensor] = []
    # NOTE: too properly measure runtimes we have to exclude compilation warmup times (first 3-5 batches)
    num_processed = 0
    total_runtime = 0.0
    encoder_runtime = 0.0
    decoder_runtime = 0.0
    with tqdm(total=args.num_samples, desc="Evaluating density estimator", unit="sample") as progress_bar:
        for batch in dataloader:
            num_in_batch = batch["tracks"].shape[0]
            num_to_evaluate = min(num_in_batch, args.num_samples - num_processed)
            scores, batch_total_runtime, batch_encoder_runtime, batch_decoder_runtime = evaluate_batch(
                batch=batch,
                encoder=encoder,
                decoder=decoder,
                sparsity=args.sparsity,
                device=device,
                max_batch_size=num_to_evaluate,
            )
            sample_scores.append(scores.detach().cpu())
            total_runtime += batch_total_runtime * num_to_evaluate
            encoder_runtime += batch_encoder_runtime * num_to_evaluate
            decoder_runtime += batch_decoder_runtime * num_to_evaluate
            num_processed += num_to_evaluate
            progress_bar.update(num_to_evaluate)
            if num_processed >= args.num_samples:
                break

    if num_processed < args.num_samples:
        raise RuntimeError(f"Dataloader produced only {num_processed} samples, requested {args.num_samples}.")
    average_energy_score = torch.nanmean(torch.cat(sample_scores)).item()
    print(f"Average exact discrete energy score over {num_processed} samples: {average_energy_score:.6f}")
    print(f"Mean total model runtime: {1000.0 * total_runtime / num_processed:.3f} ms/sample")
    print(f"Mean encoder runtime: {1000.0 * encoder_runtime / num_processed:.3f} ms/sample")
    print(f"Mean decoder runtime: {1000.0 * decoder_runtime / num_processed:.3f} ms/sample")


if __name__ == "__main__":
    arguments: Namespace = get_parser().parse_args()
    main(arguments)
