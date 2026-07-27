import random

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from typing import Annotated, TypeVar

import numpy as np
import torch
from tqdm import tqdm

from garfield.data import TrackerVideoDataModule
from garfield.model.encoder import TrajDistribEncoder, make_encoder
from garfield.model.fm_decoder import FM, make_full_decoder, make_point_decoder


torch.set_float32_matmul_precision("high")

Component = TypeVar("Component", bound=torch.nn.Module)
PCK_THRESHOLDS = (0.1, 0.01)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True, help="Path to the encoder checkpoint.")
    decoder_group = parser.add_mutually_exclusive_group(required=True)
    decoder_group.add_argument("--full", type=str, help="Path to the full trajectory decoder checkpoint.")
    decoder_group.add_argument("--point", type=str, help="Path to the point trajectory decoder checkpoint.")
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
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of predictions used for best-of-k.")
    parser.add_argument("--fm-nfe", type=int, default=20, help="Number of predictions used for best-of-k.")
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
    full_checkpoint: str | None,
    point_checkpoint: str | None,
    compile_model: bool,
) -> tuple[TrajDistribEncoder, FM]:
    encoder = load_component(encoder_checkpoint, make_encoder, "encoder").to(
        device=device,
        dtype=torch.float32,
    )
    if full_checkpoint is not None:
        decoder = load_component(full_checkpoint, make_full_decoder, "full trajectory decoder")
    elif point_checkpoint is not None:
        decoder = load_component(point_checkpoint, make_point_decoder, "point trajectory decoder")
    else:
        raise ValueError("Either --full or --point must be provided.")
    decoder = decoder.to(device=device, dtype=torch.float32)
    compile_kwargs = {
        "mode": "reduce-overhead",
        "fullgraph": True,
        "disable": not compile_model,
    }
    encoder.precompute = torch.compile(encoder.precompute, **compile_kwargs)
    encoder.forward = torch.compile(encoder.forward, **compile_kwargs)
    decoder.predict_v = torch.compile(decoder.predict_v, **compile_kwargs)
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


def aggregate_mask(
    values: Annotated[torch.Tensor, "B ..., float"],
    mask: Annotated[torch.Tensor, "B ..., bool"],
) -> Annotated[torch.Tensor, "B, float"]:
    dimensions = tuple(range(1, values.ndim))
    cast_mask = mask.to(values.dtype)
    return (values * cast_mask).sum(dim=dimensions) / cast_mask.sum(dim=dimensions)


def position_metrics(
    prediction: Annotated[torch.Tensor, "B T N 2, float"],
    target: Annotated[torch.Tensor, "B T N 2, float"],
    mask: Annotated[torch.Tensor, "B T N, bool"],
) -> dict[str, Annotated[torch.Tensor, "B, float"]]:
    l2_error = torch.linalg.vector_norm(target - prediction, dim=-1)
    metrics = {
        "epe": aggregate_mask(l2_error, mask),
        "fde": aggregate_mask(l2_error[:, -1], mask[:, -1]),
    }
    for threshold in PCK_THRESHOLDS:
        metrics[f"pck@{threshold}"] = aggregate_mask(
            (l2_error < threshold).to(l2_error.dtype),
            mask,
        )
    return metrics


def aggregate_best_of_k(
    metrics_per_prediction: list[dict[str, torch.Tensor]],
) -> dict[str, Annotated[torch.Tensor, "B, float"]]:
    result = {}
    for name in metrics_per_prediction[0]:
        values = torch.stack([metrics[name] for metrics in metrics_per_prediction], dim=-1)
        result[name] = values.max(dim=-1).values if name.startswith("pck@") else values.min(dim=-1).values
    return result


@torch.inference_mode()
def evaluate_batch(
    batch: dict[str, object],
    encoder: TrajDistribEncoder,
    decoder: FM,
    sparsity: float,
    ensemble_size: int,
    device: torch.device,
    max_batch_size: int,
    fm_nfe: int,
) -> tuple[dict[str, Annotated[torch.Tensor, "B, float"]], float, float, float]:
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

    synchronize_device(device)
    decoder_start = perf_counter()
    metrics_per_prediction = []
    for _ in range(ensemble_size):
        prediction = decoder.decode_latent(
            x_0=torch.randn_like(tracks),
            latent=latent,
            steps=fm_nfe,
        )
        metrics_per_prediction.append(position_metrics(prediction, tracks, mask=is_query))
    synchronize_device(device)
    decoder_runtime = perf_counter() - decoder_start
    total_runtime = perf_counter() - total_start

    return (
        aggregate_best_of_k(metrics_per_prediction),
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
    if args.ensemble_size <= 0:
        raise ValueError(f"--ensemble-size must be positive, got {args.ensemble_size}.")
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
        full_checkpoint=args.full,
        point_checkpoint=args.point,
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

    metric_values: dict[str, list[torch.Tensor]] = {}
    num_processed = 0
    total_runtime = 0.0
    encoder_runtime = 0.0
    decoder_runtime = 0.0
    with tqdm(total=args.num_samples, desc=f"Evaluating best-of-{args.ensemble_size}", unit="sample") as progress_bar:
        for batch in dataloader:
            num_in_batch = batch["tracks"].shape[0]
            num_to_evaluate = min(num_in_batch, args.num_samples - num_processed)
            metrics, batch_total_runtime, batch_encoder_runtime, batch_decoder_runtime = evaluate_batch(
                batch=batch,
                encoder=encoder,
                decoder=decoder,
                sparsity=args.sparsity,
                ensemble_size=args.ensemble_size,
                device=device,
                max_batch_size=num_to_evaluate,
                fm_nfe=args.fm_nfe,
            )
            for name, values in metrics.items():
                metric_values.setdefault(name, []).append(values.detach().cpu())
            total_runtime += batch_total_runtime * num_to_evaluate
            encoder_runtime += batch_encoder_runtime * num_to_evaluate
            decoder_runtime += batch_decoder_runtime * num_to_evaluate
            num_processed += num_to_evaluate
            progress_bar.update(num_to_evaluate)
            if num_processed >= args.num_samples:
                break

    if num_processed < args.num_samples:
        raise RuntimeError(f"Dataloader produced only {num_processed} samples, requested {args.num_samples}.")
    for name, values in metric_values.items():
        average = torch.nanmean(torch.cat(values)).item()
        print(f"Best-of-{args.ensemble_size} {name.upper()}: {average:.6f}")
    print(f"Mean total model runtime: {1000.0 * total_runtime / num_processed:.3f} ms/sample")
    print(f"Mean encoder runtime: {1000.0 * encoder_runtime / num_processed:.3f} ms/sample")
    print(
        f"Mean decoder runtime ({args.ensemble_size} predictions): "
        f"{1000.0 * decoder_runtime / num_processed:.3f} ms/sample"
    )


if __name__ == "__main__":
    arguments: Namespace = get_parser().parse_args()
    main(arguments)
