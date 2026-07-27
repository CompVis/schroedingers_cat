import imageio.v2 as imageio
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch

torch.set_float32_matmul_precision("high")

from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from einops import repeat
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
from typing import Annotated, TypeVar

from garfield.model.density_decoder import DensityEstimator, make_density_decoder
from garfield.model.encoder import TrajDistribEncoder, make_encoder


Component = TypeVar("Component", bound=torch.nn.Module)
GOALS_COLUMNS = ["track_id", "timestep", "x", "y"]


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default=None, help="Path to the encoder checkpoint.")
    parser.add_argument("--density", type=str, default=None, help="Path to the density decoder checkpoint.")
    parser.add_argument("--image", type=str, default="./image.png", help="Path to the input image.")
    parser.add_argument(
        "--goals", type=str, default="./goals.csv", help="Path to a CSV file which specifies the goals."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/tmp/out_density",
        help="Directory to save heatmap videos.",
    )
    parser.add_argument("--horizon", type=int, default=31, help="Number of timesteps to predict.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Power normalization gamma for density videos.")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model for faster inference.")
    parser.add_argument("--no-cam-static", action="store_true", help="Whether the camera is moving.")
    parser.add_argument("--no-cuda", action="store_true", help="Whether to disable CUDA.")
    return parser


def load_component(
    ckpt_path: str,
    make_component_fn: Callable[[], Component],
    component_name: str,
) -> Component:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        component = make_component_fn()
        component.load_state_dict(ckpt, strict=True)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load {component_name} from checkpoint '{ckpt_path}'. "
            f"Check that the file exists and contains weights compatible with {make_component_fn.__name__}. "
            f"Original error: {error}"
        ) from error
    component.eval()
    return component


def load_model(
    device: torch.device,
    dtype: torch.dtype,
    enc_ckpt: str | None,
    density_ckpt: str | None,
    compile: bool = False,
) -> tuple[TrajDistribEncoder, DensityEstimator]:
    if enc_ckpt is None:
        raise ValueError("Encoder checkpoint path must be provided with --encoder.")
    if density_ckpt is None:
        raise ValueError("Density decoder checkpoint path must be provided with --density.")

    encoder = load_component(enc_ckpt, make_encoder, "encoder").to(device=device, dtype=dtype)
    decoder = load_component(density_ckpt, make_density_decoder, "density decoder").to(device=device, dtype=dtype)
    if compile:
        encoder.forward = torch.compile(encoder.forward, fullgraph=True, mode="default")
        decoder.forward = torch.compile(decoder.forward, fullgraph=True, mode="reduce-overhead")
    return encoder, decoder


def load_image(
    image_path: str,
    size: int,
) -> Annotated[torch.Tensor, "1 H W C, float, [-1, 1]"]:
    with Image.open(image_path).convert("RGB") as pil_img:
        width, height = pil_img.size
        min_size = min(width, height)
        crop_left = (width - min_size) // 2
        crop_top = (height - min_size) // 2
        crop = pil_img.crop((crop_left, crop_top, crop_left + min_size, crop_top + min_size))
        image = np.array(crop.resize((size, size), Image.Resampling.LANCZOS))
    image = image / 127.5 - 1.0
    return torch.from_numpy(image).unsqueeze(0).float()


def load_goals(
    goals_path: str,
    nr_tracks: int,
    nr_timesteps: int,
) -> tuple[
    Annotated[torch.Tensor, "1 T N 2, float, [0, 1]"],
    Annotated[torch.Tensor, "1 T N, bool"],
]:
    goals_frame = load_goals_frame(goals_path)
    goals = torch.rand((nr_timesteps, nr_tracks, 2), dtype=torch.float32)
    is_query = torch.ones((nr_timesteps, nr_tracks), dtype=torch.bool)
    for row in range(len(goals_frame)):
        track_id = int(goals_frame.loc[row, "track_id"])
        timestep = int(goals_frame.loc[row, "timestep"])
        goals[timestep, track_id, 0] = goals_frame.loc[row, "x"]
        goals[timestep, track_id, 1] = goals_frame.loc[row, "y"]
        is_query[timestep, track_id] = False
    is_query[0] = False
    return goals.unsqueeze(0), is_query.unsqueeze(0)


def load_input(
    image_path: str,
    image_size: int,
    goals_path: str,
    nr_tracks: int,
    horizon: int,
    no_cam_static: bool,
    device: torch.device,
) -> tuple[
    Annotated[torch.Tensor, "1 H W C, float, [-1, 1]"],
    Annotated[torch.Tensor, "1 T N 2, float, [0, 1]"],
    Annotated[torch.Tensor, "1 T N, bool"],
    Annotated[torch.Tensor, "1, bool"],
]:
    image = load_image(image_path, size=image_size).to(device=device)
    goals, is_query = load_goals(goals_path, nr_tracks=nr_tracks, nr_timesteps=horizon)
    return (
        image,
        goals.to(device=device),
        is_query.to(device=device),
        torch.tensor([not no_cam_static], device=device),
    )


def load_goals_frame(goals_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(goals_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=GOALS_COLUMNS)


def get_visualized_track_ids(goals_path: str, nr_tracks: int) -> list[int]:
    goals_frame = load_goals_frame(goals_path)
    track_ids = sorted({int(track_id) for track_id in goals_frame["track_id"]})
    if not track_ids:
        return [int(np.random.randint(nr_tracks))]
    invalid_track_ids = [track_id for track_id in track_ids if track_id < 0 or track_id >= nr_tracks]
    if invalid_track_ids:
        raise ValueError(f"CSV contains track IDs outside [0, {nr_tracks - 1}]: {invalid_track_ids}")
    return track_ids


def get_conditioned_timesteps(goals_path: str, track_ids: list[int]) -> dict[int, list[int]]:
    goals_frame = load_goals_frame(goals_path)
    return {
        track_id: sorted(
            {int(timestep) for timestep in goals_frame.loc[goals_frame["track_id"] == track_id, "timestep"].tolist()}
        )
        for track_id in track_ids
    }


def get_goal_positions(goals_path: str, track_ids: list[int]) -> dict[int, np.ndarray]:
    goals_frame = load_goals_frame(goals_path)
    return {
        track_id: goals_frame.loc[goals_frame["track_id"] == track_id]
        .sort_values("timestep")[["x", "y"]]
        .to_numpy(dtype=np.float32)
        for track_id in track_ids
    }


def render_density_video(
    image: Annotated[np.ndarray, "H W 3, uint8"],
    heatmaps: Annotated[np.ndarray, "T H_grid W_grid, float"],
    conditioned_timesteps: list[int],
    goal_positions: Annotated[np.ndarray, "P 2, float, [0, 1]"] | None = None,
    gamma: float = 0.4,
    opacity: float = 0.7,
) -> Annotated[np.ndarray, "T H W_total 3, uint8"]:
    if gamma <= 0:
        raise ValueError(f"Power normalization gamma must be positive, got {gamma}.")
    density_min = float(heatmaps.min())
    density_max = float(heatmaps.max())
    if density_max <= density_min:
        density_max = density_min + np.finfo(np.float32).eps
    norm = mpl.colors.PowerNorm(gamma=gamma, vmin=density_min, vmax=density_max)
    nr_timesteps = heatmaps.shape[0]
    invalid_timesteps = [timestep for timestep in conditioned_timesteps if timestep < 0 or timestep >= nr_timesteps]
    if invalid_timesteps:
        raise ValueError(f"Conditioned timesteps outside [0, {nr_timesteps - 1}]: {invalid_timesteps}")

    height, width, _ = image.shape
    figure = Figure(figsize=(8.0, 4.0), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    grid_spec = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.06],
        height_ratios=[1.0, 0.14],
        wspace=0.08,
        hspace=0.08,
    )
    overlay_axis = figure.add_subplot(grid_spec[0, 0])
    heatmap_axis = figure.add_subplot(grid_spec[0, 1])
    colorbar_axis = figure.add_subplot(grid_spec[0, 2])
    timeline_axis = figure.add_subplot(grid_spec[1, :2])

    overlay_axis.imshow(image, interpolation="nearest", zorder=0)
    if goal_positions is not None:
        for start, end in zip(goal_positions[:-1], goal_positions[1:], strict=True):
            overlay_axis.arrow(
                start[0] * width,
                start[1] * height,
                (end[0] - start[0]) * width,
                (end[1] - start[1]) * height,
                color="deepskyblue",
                width=1.0,
                head_width=12.0,
                head_length=12.0,
                length_includes_head=True,
                zorder=1,
            )
    overlay_plot = overlay_axis.imshow(
        heatmaps[0],
        cmap="viridis",
        norm=norm,
        alpha=np.asarray(norm(heatmaps[0])) * opacity,
        interpolation="nearest",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        zorder=2,
    )
    overlay_axis.set_xlim(-0.5, width - 0.5)
    overlay_axis.set_ylim(height - 0.5, -0.5)
    overlay_axis.set_axis_off()

    heatmap_plot = heatmap_axis.imshow(
        heatmaps[0],
        cmap="viridis",
        norm=norm,
        interpolation="nearest",
    )
    heatmap_axis.set_axis_off()
    colorbar = figure.colorbar(heatmap_plot, cax=colorbar_axis)
    colorbar.set_label("Density (power scaled)")

    timeline_axis.axhline(0.5, color="0.65", linewidth=2.0)
    if conditioned_timesteps:
        timeline_axis.scatter(
            conditioned_timesteps,
            [0.5] * len(conditioned_timesteps),
            color="deepskyblue",
            edgecolors="black",
            marker="D",
            s=35,
            zorder=3,
        )
    timeline_cursor = timeline_axis.axvline(0, color="red", linewidth=2.0)
    timeline_axis.set_xlim(-0.5, nr_timesteps - 0.5)
    timeline_axis.set_ylim(0.0, 1.0)
    timeline_axis.set_xticks(sorted({0, nr_timesteps - 1, *conditioned_timesteps}))
    timeline_axis.set_xlabel("Timestep (cyan diamonds: conditions; red line: current)")
    timeline_axis.get_yaxis().set_visible(False)
    timeline_axis.spines[["left", "right", "top"]].set_visible(False)

    frames: list[np.ndarray] = []
    for timestep, heatmap in enumerate(heatmaps):
        overlay_plot.set_data(heatmap)
        overlay_plot.set_alpha(np.asarray(norm(heatmap)) * opacity)
        heatmap_plot.set_data(heatmap)
        timeline_cursor.set_xdata([timestep, timestep])
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return np.stack(frames, axis=0)


def save_density_videos(
    density: Annotated[torch.Tensor, "1 T N H W 1, float"],
    image: Annotated[torch.Tensor, "1 H W 3, float, [-1, 1]"],
    track_ids: list[int],
    conditioned_timesteps: dict[int, list[int]],
    goal_positions: dict[int, np.ndarray],
    output_dir: str,
    gamma: float,
) -> None:
    density_array = density[0, ..., 0].detach().float().cpu().numpy()
    # Density targets use (x, y), while image arrays are indexed as (row=y, column=x).
    density_array = density_array.swapaxes(-2, -1)
    image_array = image[0].detach().float().cpu().numpy()
    image_array = ((image_array + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for track_id in track_ids:
        frames = render_density_video(
            image=image_array,
            heatmaps=density_array[:, track_id],
            conditioned_timesteps=conditioned_timesteps[track_id],
            goal_positions=goal_positions[track_id],
            gamma=gamma,
        )
        video_path = output_path / f"track_{track_id:02d}_density.mp4"
        imageio.mimsave(video_path, frames, fps=8, macro_block_size=1)
        print(f"Saved density video to {video_path}", flush=True)


@torch.no_grad()
def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dtype = torch.float32
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    encoder, decoder = load_model(
        device=device,
        dtype=dtype,
        enc_ckpt=args.encoder,
        density_ckpt=args.density,
        compile=args.compile,
    )
    nr_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    nr_parameters += sum(parameter.numel() for parameter in decoder.parameters())
    print(f"Loaded encoder and density decoder with {nr_parameters:,} parameters.", flush=True)

    nr_tracks = encoder.track_identity_table.nr_tracks
    image, goals, is_query, is_cam_static = load_input(
        image_path=args.image,
        image_size=encoder.image_encoder.resize,
        goals_path=args.goals,
        nr_tracks=nr_tracks,
        horizon=args.horizon,
        no_cam_static=args.no_cam_static,
        device=device,
    )
    precomputed = encoder.precompute(image, tracks=goals)
    latent = encoder(
        goals,
        is_query,
        is_cam_static,
        precomputed_kwargs=precomputed,
    ).clone()

    batch_size, horizon, nr_tracks, _ = goals.shape
    t_idx = repeat(
        torch.arange(horizon, device=device, dtype=torch.float32),
        "T -> B T N",
        B=batch_size,
        N=nr_tracks,
    )
    t_idx = t_idx / horizon
    x_idx = repeat(goals[:, 0, :, 0], "B N -> B T N", T=horizon)
    y_idx = repeat(goals[:, 0, :, 1], "B N -> B T N", T=horizon)
    density = decoder.decode_latent(
        latent=latent,
        split_bs=1024,
        t_idx=t_idx,
        x_idx=x_idx,
        y_idx=y_idx,
    )
    print(f"Density output shape: {tuple(density.shape)}", flush=True)
    track_ids = get_visualized_track_ids(args.goals, nr_tracks=nr_tracks)
    conditioned_timesteps = get_conditioned_timesteps(args.goals, track_ids=track_ids)
    goal_positions = get_goal_positions(args.goals, track_ids=track_ids)
    save_density_videos(
        density=density,
        image=image,
        track_ids=track_ids,
        conditioned_timesteps=conditioned_timesteps,
        goal_positions=goal_positions,
        output_dir=args.output,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    args: Namespace = get_parser().parse_args()
    main(args)
