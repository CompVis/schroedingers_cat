import logging
import os
import random

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from pydoc import locate

import numpy as np
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from garfield.data import TrackerVideoDataModule
from garfield.model.density_decoder import make_density_decoder
from garfield.model.encoder import make_encoder
from garfield.model.fm_decoder import make_full_decoder, make_point_decoder
from garfield.model.main import GARFIELD, SparsityScheduler


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--stage", choices=("point", "full", "density"), required=True)
    parser.add_argument("--encoder", type=str, default=None, help="Optional encoder checkpoint.")
    parser.add_argument("--decoder", type=str, default=None, help="Optional decoder checkpoint.")
    parser.add_argument("--tarbase", type=str, required=True, help="Directory containing training tar shards.")
    parser.add_argument("--shards", type=str, default=None, help="Shard glob relative to --tarbase.")
    parser.add_argument("--out-dir", type=str, default="./runs", help="Output directory.")
    parser.add_argument("--resume", type=str, default=None, help="Training checkpoint produced by this script.")
    parser.add_argument("--dtype", type=str, default="torch.bfloat16", help="Training precision.")
    parser.add_argument("--batch-size", type=int, default=16, help="Local batch size per device.")
    parser.add_argument("--nr-steps", type=int, required=True, help="Number of optimizer steps.")
    parser.add_argument("--nr-sparsity-steps", type=int, default=50000, help="Number of optimizer steps.")
    parser.add_argument("--nr-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adamw-decay", type=float, default=0.01)
    parser.add_argument("--ckpt-freq", type=int, default=10000)
    parser.add_argument("--load-optim", action="store_true")
    parser.add_argument("--load-sched", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--horizon", type=int, default=31)
    parser.add_argument("--num-tracks", type=int, default=64)
    parser.add_argument("--sparsity-start", type=float, default=0.5)
    parser.add_argument("--sparsity-end", type=float, default=0.99)
    parser.add_argument("--density-t-samples", type=int, default=4)
    parser.add_argument("--density-n-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: Namespace) -> None:
    if args.nr_steps <= 0:
        raise ValueError("--nr-steps must be positive.")
    if args.batch_size <= 0 or args.nr_workers < 0:
        raise ValueError("--batch-size must be positive and --nr-workers non-negative.")
    if args.warmup_steps <= 0 or args.ckpt_freq <= 0:
        raise ValueError("--warmup-steps and --ckpt-freq must be positive.")
    if not 0.0 <= args.sparsity_start <= 1.0 or not 0.0 <= args.sparsity_end <= 1.0:
        raise ValueError("Sparsity values must be in [0, 1].")
    if args.stage != "point" and args.encoder is None and args.resume is None:
        raise ValueError(f"The frozen-encoder '{args.stage}' stage requires --encoder or --resume.")
    if not Path(args.tarbase).is_dir():
        raise ValueError(f"--tarbase must be an existing directory, got '{args.tarbase}'.")


def load_weights(module: torch.nn.Module, checkpoint_path: str, component_name: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    try:
        module.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load {component_name} checkpoint '{checkpoint_path}': {error}"
        ) from error


def make_model(args: Namespace) -> GARFIELD:
    encoder = make_encoder()
    if args.encoder is not None:
        load_weights(encoder, args.encoder, "encoder")

    if args.stage == "point":
        decoder = make_point_decoder()
    elif args.stage == "full":
        decoder = make_full_decoder()
    elif args.stage == "density":
        decoder = make_density_decoder()
        decoder.sampling_kwargs = {
            "T": args.density_t_samples,
            "N": args.density_n_samples,
            "is_randperm": True,
        }
    else:
        raise ValueError(f"Unknown stage '{args.stage=}'.")
    if args.decoder is not None:
        load_weights(decoder, args.decoder, f"{args.stage} decoder")

    return GARFIELD(
        encoder=encoder,
        decoder=decoder,
        sparsity_scheduler=SparsityScheduler(
            start_sparsity=args.sparsity_start,
            end_sparsity=args.sparsity_end,
            total_steps=args.nr_sparsity_steps,
        ),
        freeze_encoder=args.stage != "point",
    )


def make_data_module(args: Namespace, image_size: int) -> TrackerVideoDataModule:
    data_config = {
        "tar_base": args.tarbase,
        "shards": args.shards,
        "batch_size": args.batch_size,
        "num_workers": args.nr_workers,
        "shuffle": 1000,
        "simple_heuristic": True,
        "filter_static_camera": False,
        "decode_kwargs": {
            "track_key": "tracks_yx",
            "image_size": image_size,
            "num_steps": args.horizon + 1,
            "num_tracks": args.num_tracks,
            "frame_skip": 1,
            "center_crop": True,
            "certainty_threshold": 0.6,
            "return_full_sequence": False,
            "return_video_without_cutting": False,
            "static_camera_flow_mag_threshold": 0.0002,
            "static_camera_fraction_threshold": 0.45,
        },
    }
    return TrackerVideoDataModule(train=data_config, validation=data_config)


def compute_step(model: torch.nn.Module, batch: dict[str, object], device: torch.device) -> torch.Tensor:
    required = ("cond_frame", "cond_frame_idx", "tracks", "is_camera_static")
    tensors = {}
    for key in required:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor batch['{key}'], got {type(value).__name__}.")
        tensors[key] = value.to(device=device, non_blocking=True)
    tensors["cond_frame"] = tensors["cond_frame"].float()
    tensors["tracks"] = tensors["tracks"].float()
    loss = model(**tensors)
    return loss.mean()


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: GARFIELD,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    step_dir = os.path.join(checkpoint_dir, f"step-{step}")
    os.makedirs(step_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(step_dir, "encoder.pt"))
    torch.save(model.decoder.state_dict(), os.path.join(step_dir, "decoder.pt"))
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
        },
        os.path.join(step_dir, "train.pt"),
    )


def main(args: Namespace) -> None:
    validate_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_distributed:
        dist.init_process_group()
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = locate(args.dtype)

    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%H-%M-%S"))
    out_base = os.path.join(args.out_dir, datetime.now().strftime("%Y-%m-%d"), run_id)
    checkpoint_dir = os.path.join(out_base, "ckpt")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logging.getLogger("garfield.train")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = make_model(args).to(device)
    first_step = 0
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        first_step = int(resume_checkpoint["step"]) + 1
    model.train()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.adamw_decay)
    scheduler = LinearLR(
        optimizer,
        start_factor=min(1.0, 1e-8 / args.lr),
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    if resume_checkpoint is not None and args.load_optim:
        optimizer.load_state_dict(resume_checkpoint["optim"])
    if resume_checkpoint is not None and args.load_sched:
        scheduler.load_state_dict(resume_checkpoint["sched"])

    training_model: torch.nn.Module = torch.compile(model, mode="max-autotune") if args.compile else model
    if is_distributed:
        training_model = DDP(training_model, device_ids=[local_rank])
    data = make_data_module(args, image_size=model.encoder.image_encoder.resize)
    train_loader = data.train_dataloader()

    if rank == 0:
        logger.info(
            "Training stage=%s, freeze_encoder=%s, trainable_parameters=%d",
            args.stage,
            model.freeze_encoder,
            sum(parameter.numel() for parameter in trainable_parameters),
        )
    if is_distributed:
        dist.barrier()

    progress = tqdm(
        total=args.nr_steps,
        initial=first_step,
        desc=f"Training {args.stage}",
        disable=rank != 0,
        unit="step",
    )
    for batch in train_loader:
        if first_step >= args.nr_steps:
            break
        with torch.autocast(device_type=device.type, dtype=dtype):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_step(training_model, batch, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
        first_step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss.detach().item():.5f}")

        should_save = first_step % args.ckpt_freq == 0 or first_step == args.nr_steps
        if should_save and rank == 0:
            save_checkpoint(checkpoint_dir, first_step, model, optimizer, scheduler)
            logger.info("Saved checkpoint at step %d to %s", first_step, checkpoint_dir)
    progress.close()

    if is_distributed:
        dist.barrier()
    if rank == 0:
        logger.info("Training finished at step %d", first_step)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    arguments = get_parser().parse_args()
    try:
        main(arguments)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
