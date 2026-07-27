import io
import os
import tempfile
from pathlib import Path
from typing import Any

import click
import cv2
import imageio
import numpy as np
import webdataset as wds

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tapnext_mplconfig_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="tapnext_cache_"))

from tapnet.utils.viz_utils import plot_tracks_v2  # noqa: E402


def find_value(sample: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in sample:
            return sample[key]

    candidate_suffixes = tuple(candidates)
    for key, value in sample.items():
        if key.endswith(candidate_suffixes):
            return value

    raise KeyError(f"Could not find any of {candidates}. Sample keys: {sorted(sample.keys())}")


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, bytes):
        return np.load(io.BytesIO(value))
    return np.asarray(value)


def decode_mp4(data: bytes) -> tuple[np.ndarray, float]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Could not open MP4 payload from shard sample.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()

    if len(frames) == 0:
        raise ValueError("MP4 payload contained no decodable frames.")

    if not fps or np.isnan(fps) or fps < 1e-3:
        fps = 12.0

    return np.stack(frames, axis=0), float(fps)


def load_sample(shard: Path, sample_index: int) -> dict[str, Any]:
    dataset = wds.DataPipeline(
        wds.shardlists.SimpleShardList(str(shard)),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode(),
    )

    for index, sample in enumerate(dataset):
        if index == sample_index:
            return sample

    raise IndexError(f"Shard {shard} does not contain sample index {sample_index}.")


def tracks_to_plot_points(tracks_yx: np.ndarray, height: int, width: int) -> np.ndarray:
    coords_yx = (tracks_yx.astype(np.float32) + 1.0) * 0.5
    coords_yx = coords_yx * np.array([height, width], dtype=np.float32)
    points_xy = np.stack([coords_yx[..., 1], coords_yx[..., 0]], axis=-1)
    return np.transpose(points_xy, (1, 0, 2))


def select_tracks(
    points: np.ndarray,
    occluded: np.ndarray,
    query_frame_index: np.ndarray | None,
    max_tracks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_tracks <= 0 or points.shape[0] <= max_tracks:
        return points, occluded, query_frame_index

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points.shape[0], size=max_tracks, replace=False))
    query_frame_index = None if query_frame_index is None else query_frame_index[indices]
    return points[indices], occluded[indices], query_frame_index


@click.command()
@click.option("--shard", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--sample-index", type=int, default=0, show_default=True)
@click.option("--video-key", type=str, default="video.mp4", show_default=True)
@click.option("--max-tracks", type=int, default=256, show_default=True, help="Use <= 0 to draw all tracks.")
@click.option("--point-size", type=int, default=8, show_default=True)
@click.option("--visibility-threshold", type=float, default=0.0, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    shard: Path,
    output: Path,
    sample_index: int,
    video_key: str,
    max_tracks: int,
    point_size: int,
    visibility_threshold: float,
    seed: int,
) -> None:
    sample = load_sample(shard, sample_index)

    video_bytes = find_value(sample, [video_key, "mp4"])
    if not isinstance(video_bytes, bytes):
        raise TypeError(f"Expected MP4 bytes for {video_key}, got {type(video_bytes)}")

    rgb, fps = decode_mp4(video_bytes)
    tracks_yx = as_numpy(find_value(sample, ["tracks_yx.npy", "tracks_yx"]))
    visible_logits = as_numpy(find_value(sample, ["logits_visible.npy", "logits_visible"]))

    if tracks_yx.ndim != 3 or tracks_yx.shape[-1] != 2:
        raise ValueError(f"Expected tracks_yx shape [T, Q, 2], got {tracks_yx.shape}")
    if visible_logits.shape != tracks_yx.shape[:2]:
        raise ValueError(f"Expected logits_visible shape {tracks_yx.shape[:2]}, got {visible_logits.shape}")
    if rgb.shape[0] != tracks_yx.shape[0]:
        raise ValueError(f"Video has {rgb.shape[0]} frames but tracks have {tracks_yx.shape[0]} frames.")

    try:
        query_frame_index = as_numpy(find_value(sample, ["query_frame_index.npy", "query_frame_index"])).astype(int)
    except KeyError:
        query_frame_index = None

    points = tracks_to_plot_points(tracks_yx, height=rgb.shape[1], width=rgb.shape[2])
    occluded = np.transpose(visible_logits <= visibility_threshold, (1, 0))

    if query_frame_index is not None:
        time_indices = np.arange(rgb.shape[0])[None, :]
        occluded = np.logical_or(occluded, time_indices < query_frame_index[:, None])

    points, occluded, _ = select_tracks(points, occluded, query_frame_index, max_tracks=max_tracks, seed=seed)
    video = plot_tracks_v2(rgb=rgb, points=points, occluded=occluded, point_size=point_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, video, fps=fps, macro_block_size=1)
    print(f"Wrote {output} from sample {sample.get('__key__', sample_index)} with {points.shape[0]} tracks.")


if __name__ == "__main__":
    main()
