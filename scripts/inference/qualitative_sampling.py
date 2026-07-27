# load model (encoder + FM decoder), compile option
# load image
# load constraints / (sub)goals
# embed into latent
# feed into FM decoder
# visualize spaghetti

import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from argparse import ArgumentParser, Namespace
from einops import rearrange, repeat
from typing import Annotated, Callable
from PIL import Image

from garfield.model.encoder import make_encoder, TrajDistribEncoder
from garfield.model.fm_decoder import make_point_decoder, make_full_decoder, FM

# rm -rf garfield/model/__pycache__ && rm -rf scripts/inference/__pycache__ && python -m scripts.inference.qualitative_sampling --encoder tmp/ckpt/encoder.pt --full tmp/ckpt/full.pt --image tmp/qual_input/tigerhunt.png --goals tmp/qual_input/goals.csv

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default=None, help="Path to the model checkpoint.",)
    parser.add_argument("--full", type=str, default=None, help="Path to the full model checkpoint.",)
    parser.add_argument("--point", type=str, default=None, help="Path to the point model checkpoint.",)
    parser.add_argument("--image", type=str, default="./image.png", help="Path to the input image.",)
    parser.add_argument("--goals", type=str, default="./goals.csv", help="Path to a CSV file which specifies the goals.",)
    parser.add_argument("--output", type=str, default="./tmp/out", help="Directory to save the output to.",)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.",)
    parser.add_argument("--horizon", type=int, default=31, help="Number of timesteps to predict.",)
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model for faster inference.",)
    parser.add_argument("--no-cam-static", action="store_true", help="Whether the camera is moving.",)
    parser.add_argument("--no-cuda", action="store_true", help="Whether to disable CUDA.",)
    return parser

def load_component(ckpt_path: str, make_component_fn: Callable[[], torch.nn.Module],) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    component = make_component_fn()
    component.load_state_dict(ckpt)
    component.eval()
    return component

def load_model(
    device: torch.device,
    dtype: torch.dtype,
    enc_ckpt: str,
    full_ckpt: str | None = None,
    point_ckpt: str | None = None,
    compile: bool = False,
) -> tuple[TrajDistribEncoder, FM]:
    assert enc_ckpt is not None, "Encoder checkpoint path must be provided."
    encoder: TrajDistribEncoder = load_component(enc_ckpt, make_encoder).to(device=device, dtype=dtype)
    if full_ckpt is not None:
        decoder = load_component(full_ckpt, make_full_decoder).to(device=device, dtype=dtype)
    elif point_ckpt is not None:
        decoder = load_component(point_ckpt, make_point_decoder).to(device=device, dtype=dtype)
    else:
        raise ValueError("Either full or point decoder checkpoint path must be provided.")
    if compile:
        compile_kwargs = {"fullgraph": True, "mode": "reduce-overhead"}
        encoder.precompute = torch.compile(encoder.precompute, **compile_kwargs)
        encoder.forward = torch.compile(encoder.forward, **compile_kwargs)
        decoder.decode_latent = torch.compile(decoder.decode_latent, **compile_kwargs)
    return encoder, decoder

def load_image(image_path: str, size: int) -> Annotated[torch.Tensor, "1 H W C, float, [-1, 1]"]:

    with Image.open(image_path).convert("RGB") as pil_img:
        W, H = pil_img.size
        min_size = min(W, H)
        crop_left = (W - min_size) // 2
        crop_right = crop_left + min_size
        crop_top = (H - min_size) // 2
        crop_bottom = crop_top + min_size
        crop = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        pil_img = crop.resize((size, size), Image.LANCZOS)
        img = np.array(crop)
    img = img / 127.5 - 1.0  # normalize to [-1, 1]
    tensor = torch.from_numpy(img)
    tensor = rearrange(tensor, "H W C -> 1 H W C").float()
    return tensor

def load_goals(goals_path: str, nr_tracks: int, nr_timesteps: int) -> tuple[Annotated[torch.Tensor, "1 T N 2, float, [0, 1]"], Annotated[torch.Tensor, "1 T N, bool"]]:
    df = pd.read_csv(goals_path)
    goals = torch.rand((nr_timesteps, nr_tracks, 2), dtype=torch.float32)
    is_query = torch.ones((nr_timesteps, nr_tracks), dtype=torch.bool)
    nr_rows = len(df.shape)
    for row in range(nr_rows):
        track_id = df.loc[row, "track_id"]
        timestep = df.loc[row, "timestep"]
        x = df.loc[row, "x"]
        y = df.loc[row, "y"]
        goals[timestep, track_id, 0] = x
        goals[timestep, track_id, 1] = y
        is_query[timestep, track_id] = False
    is_query[0] = False  # first timestep is always conditioned
    goals = repeat(goals, "T N D -> 1 T N D")
    is_query = repeat(is_query, "T N -> 1 T N")
    nr_goals = (~is_query).sum().item()
    return goals, is_query

def load_input(
    img_path: str,
    img_size: int,
    goals_path: str,
    nr_tracks: int,
    horizon: int,
    no_cam_static: bool,
    device: torch.device,
    dtype: torch.dtype,
):
    img = load_image(img_path, size=img_size).to(device, dtype=dtype)
    goals, is_query = load_goals(goals_path, nr_tracks=nr_tracks, nr_timesteps=horizon)
    goals = goals.to(device, dtype=dtype)
    is_query = is_query.to(device)
    is_cam_static = torch.tensor([not no_cam_static], device=device)
    return img, goals, is_query, is_cam_static


def setup_hsv_colormap(
    nr_tracks: int,
    default_sat: float = 1.0,
    default_val: float = 1.0,
) -> list[tuple[float, float, float]]:
    result = []
    for track_idx in range(nr_tracks):
        hue = track_idx / nr_tracks
        saturation = default_sat
        value = default_val
        hsv = (hue, saturation, value)
        result.append(hsv)
    return result

def plt_goals(
    image: Annotated[np.ndarray, "H W C, uint8"],
    goals: Annotated[np.ndarray, "T N 2, float, [0, 1]"],
    is_query: Annotated[np.ndarray, "T N, bool"],
    color_map: list[tuple[float, float, float]],
    path: str | None = None,
) -> Annotated[np.ndarray, "H W rgba, uint8"] | None:
    fig, ax = plt.subplots()
    H,W,_ = image.shape
    ax.imshow(image)
    T,N,_ = goals.shape
    for n in range(N):
        hue, saturation, value = color_map[n]
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        nr_known = (~is_query[:, n]).astype(np.int32).sum()
        assert is_query[0, n] == False, f"Track {n} must have a known goal at timestep 0 (the start position)."
        assert nr_known >= 0, f"Every track has at least one known goal (the start position)."
        if nr_known == 1:
            continue
        prev_t = 0
        for t in range(1, T):
            if not is_query[t, n]:
                # ax.plot(
                #     [goals[prev_t, n, 0] * W, goals[t, n, 0] * W],
                #     [goals[prev_t, n, 1] * H, goals[t, n, 1] * H],
                #     color=rgb,
                #     linewidth=2.0,
                # )
                ax.arrow(
                    goals[prev_t, n, 0] * W,
                    goals[prev_t, n, 1] * H,
                    (goals[t, n, 0] - goals[prev_t, n, 0]) * W,
                    (goals[t, n, 1] - goals[prev_t, n, 1]) * H,
                    color=rgb,
                    width=1.0,
                    head_width=20.0,
                    head_length=20.0,
                    length_includes_head=True,
                )
                prev_t = t
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.axis("off")
    ax.margins(0)
    if path is not None:
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.cla()
    plt.clf()
    plt.close()

def plt_tracks(
    image: Annotated[np.ndarray, "H W C, uint8"],
    tracks: Annotated[np.ndarray, "T N 2, float, [0, H/W]"],
    color_map: list[tuple[float, float, float]],
    path: str | None = None,
) -> Annotated[np.ndarray, "H W rgba, uint8"] | None:
    fig, ax = plt.subplots()
    H,W,_ = image.shape
    ax.imshow(image)
    T = tracks.shape[0]
    for track_idx, hsv_color in enumerate(color_map):
        hue, saturation, value = hsv_color
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        track_x = tracks[:, track_idx, 0]
        track_y = tracks[:, track_idx, 1]
        ax.plot(track_x, track_y, color=rgb, linewidth=2.0)
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.axis("off")
    ax.margins(0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.cla()
    plt.clf()
    plt.close()

@torch.no_grad()
def main(args: Namespace) -> None:
    # basic checks
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    
    encoder, decoder = load_model(
        device=device,
        dtype=dtype,
        enc_ckpt=args.encoder,
        full_ckpt=args.full,
        point_ckpt=args.point,
        compile=args.compile,
    )
    nr_tracks = encoder.track_identity_table.nr_tracks
    img, goals, is_query, is_cam_static = load_input(
        img_path=args.image,
        img_size=encoder.image_encoder.resize,
        goals_path=args.goals,
        nr_tracks=nr_tracks,
        horizon=args.horizon,
        no_cam_static=args.no_cam_static,
        device=device,
        dtype=dtype,
    )

    precomputed = encoder.precompute(img, tracks=goals)
    precomputed_keys = list(precomputed.keys())
    precomputed_keys.sort()

    latent = encoder.forward(goals, is_query, is_cam_static, precomputed_kwargs=precomputed)
    noise = torch.randn_like(goals)
    sample = decoder.decode_latent(x_0=noise, latent=latent,)
    sample_path = os.path.join(args.output, "sample.npy")
    np_sample = sample.cpu().numpy()
    np.save(sample_path, np_sample)
    print(f"Saved sample to {sample_path=}", flush=True)

    hsv_map = setup_hsv_colormap(nr_tracks=nr_tracks)
    np_img = img[0].cpu().numpy()
    np_img = (np_img + 1.0) * 127.5
    np_img = np_img.astype(np.uint8)
    H, W, _ = np_img.shape
    np_sample = np_sample[0].clip(0, 1)
    np_sample[..., 0] *= W
    np_sample[..., 1] *= H

    goals_path = os.path.join(args.output, "goals.png")
    plt_goals(
        image=np_img,
        goals=goals[0].cpu().clip(0,1).numpy(),
        is_query=is_query[0].cpu().numpy(),
        color_map=hsv_map,
        path=goals_path,
    )
    print(f"Saved visualization to {goals_path=}", flush=True)

    vis_path = os.path.join(args.output, "sample.png")
    plt_tracks(
        image=np_img,
        tracks=np_sample,
        color_map=hsv_map,
        path=vis_path,
    )
    print(f"Saved visualization to {vis_path=}", flush=True)



if __name__ == "__main__":
    args: Namespace = get_parser().parse_args()
    main(args)
