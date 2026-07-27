# minimal dataloader: videos, tracks, filtering params

import os
import shutil

from pathlib import Path
import io
import random
import json

import torch
import numpy as np
import torchvision.transforms as T
try:
    import torchvision.transforms.v2 as TVT
    v2_available = True
except ImportError:
    TVT = T
    v2_available = False
import av
import einops
import webdataset as wds
from jaxtyping import Float, Bool
from typing import Any
import cv2
from functools import partial
import glob
from omegaconf import OmegaConf, ListConfig

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}  # remove keys with "__"

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.from_numpy(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.from_numpy(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])
    return result


def augment(
    clip,
    interpolation=TVT.InterpolationMode.BICUBIC,
    size=512,
    center_crop=False,
):
    if v2_available:
        # make shorter side to size!
        clip = TVT.functional.resize(clip, size, interpolation=interpolation, antialias=True)


        # normalize
        clip = (clip - 0.5) / 0.5
        clip = clip.clamp(-1.0, 1.0)  # to prevent values outside from [-1,1] in bicubic mode
        return clip
    else:
        if clip.dtype == torch.uint8:
            clip = clip.float().div_(255.0)
        else:
            clip = clip.float()

        clip = TVT.functional.resize(
            clip,
            size,
            interpolation=interpolation,
            antialias=True,
        )

        c = clip.shape[-3]
        clip = TVT.Normalize(mean=[0.5] * c, std=[0.5] * c)(clip)
        clip = clip.clamp(-1.0, 1.0)
        return clip




from webdataset.filters import reraise_exception, pipelinefilter


def _map_many(data, f, handler=reraise_exception):
    """Version of wds.map() that ."""
    for sample in data:
        try:
            results = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        for i, r in enumerate(results):
            if r is None:
                continue
            if isinstance(sample, dict) and isinstance(r, dict):
                r["__key__"] = f"{sample.get('__key__', )}-{i}"
            yield r


map_many = pipelinefilter(_map_many)

def decode_npy(b: bytes):
    with io.BytesIO(b) as f:
        return np.load(f, allow_pickle=True)

def compute_homography(p0, p1, ransac_thresh=0.01):
    """
    Compute homography using two sets of points p0 and p1 (both of shape (N, 2))
    with a minimum of 4 points, using RANSAC.

    Parameters:
      - p0: Source points, shape (N,2)
      - p1: Destination points, shape (N,2)
      - ransac_thresh: RANSAC reprojection threshold (assumed in normalized units)

    Returns:
      - H: 3x3 homography matrix (or None if estimation fails)
      - mask: Inlier mask (or None if estimation fails)
    """
    if p0.shape[0] < 4:  # Need at least 4 points for a valid homography
        return None, None
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, ransac_thresh)
    return H, mask


def check_static_camera_first_last(
    sparse_tracks, visibility, ransac_thresh=0.01, homography_id_thresh=0.01, inlier_ratio_thresh=0.8
):
    T = sparse_tracks.shape[0]
    valid_transforms = []
    num_valid = 0
    total_samples = 0

    sample_interval = T - 1

    # Loop through frames by sample_interval steps.
    for i in range(0, T - sample_interval, sample_interval):
        # Select keypoints visible in both frame i and frame i+sample_interval.
        common_visible = visibility[i] & visibility[i + sample_interval]
        if np.sum(common_visible) < 4:
            continue  # Skip if not enough keypoints are visible in both frames.

        p0 = sparse_tracks[i][common_visible][:, [1,0]]
        p1 = sparse_tracks[i + sample_interval][common_visible][:, [1,0]]

        H, mask = compute_homography(p0, p1, ransac_thresh)
        # If homography estimation failed, skip this pair.
        if H is None or mask is None:
            continue

        total_samples += 1
        # Compute the inlier ratio from the mask.
        inlier_ratio = float(np.sum(mask)) / mask.size

        # Normalize the homography so that H[2, 2] == 1.
        H_normalized = H / H[2, 2]
        identity = np.eye(3, dtype=np.float32)
        diff = np.linalg.norm(H_normalized - identity)

        # Accept as static if the homography is close to identity and the inlier ratio is high.
        if diff < homography_id_thresh and inlier_ratio > inlier_ratio_thresh:
            valid_transforms.append((i, i + sample_interval, diff, inlier_ratio))
            num_valid += 1

    # Declare the camera static if a majority (e.g., 80%) of the sampled frame pairs are static.
    camera_static = (total_samples >= 0) and (num_valid >= total_samples * 0.8)
    return camera_static, valid_transforms

class TrackerVideoDataModule:
    def __init__(
        self,
        train: dict[str, Any],
        validation: dict[str, Any],
    ):
        super().__init__()
        self.train = train
        self.validation = validation

    def extract_training_sample(
        self,
        sample: dict[str, torch.Tensor],
        frame_skip: int,
        num_tracks: int,
        num_steps: int,
        min_motion_threshold: float = 0.0,
        visibility_threshold: float = 0.5,
        certainty_threshold: float = 0.5,
        static_camera_flow_mag_threshold: float = 0.0007,
        static_camera_fraction_threshold: float = 0.5,
        allow_invisible_track_ends: bool = False,
    ) -> dict[str, torch.Tensor]:
        for k in [
            "visibility",
            "tracks",
            "times",
            'certainty'
        ]:
            assert k in sample.keys(), f"{k=} {sample.keys()=}"
        sample_key = sample["__key__"]
        try:
            visibility: Bool[torch.Tensor, "t n_t"] = sample["visibility"]
            tracks: Float[torch.Tensor, "t n_t 2"] = sample["tracks"]
            certainty: Float[torch.Tensor, "t n_t"] = sample["certainty"]

            active = (visibility > visibility_threshold) & (certainty > certainty_threshold)
            if active.shape[-1] == 1:
                active = active.squeeze(dim=-1)

            track_in_frame: Bool[torch.Tensor, "t n_t"] = (
                (tracks[..., 0] >= 0) & (tracks[..., 0] <= 1) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= 1)
            )
            visible_and_in_frame: Bool[torch.Tensor, "t n_t"] = active & track_in_frame
            valid_start_frames = (
                visible_and_in_frame.int().sum(dim=1) >= num_tracks
            )  # Ones that have at least `num_tracks` visible tracks
            if not valid_start_frames.any():
                return {"valid": False}

            valid_start_frames = valid_start_frames & (
                torch.arange(valid_start_frames.size(0)) <= valid_start_frames.size(0) - 1 - ((num_steps - 1) * frame_skip)
            )
            if not valid_start_frames.any():
                return {"valid": False}
            i_start = valid_start_frames.nonzero()[torch.randint(valid_start_frames.sum(), (1,))].squeeze()
            i_end = i_start + (num_steps * frame_skip)
            i_last = i_start + (num_steps - 1) * frame_skip

            valid_tracks_mask = visible_and_in_frame[i_start]
            if not allow_invisible_track_ends:
                valid_tracks_mask &= active[i_last]
            """if not self.allow_out_of_frame_track_ends:    # NOTE: I think this case is already covered by our updated visibility after cropping
                valid_tracks_mask &= track_in_frame[i_last]"""

            # Select random valid tracks
            valid_tracks = valid_tracks_mask.nonzero().flatten()
            if len(valid_tracks) < num_tracks:
                return {"valid": False}
            selected_track_idxs = valid_tracks[torch.randperm(len(valid_tracks))]
            
            pos: Float[torch.Tensor, "t l 2"] = tracks[i_start : i_last+1 : frame_skip]
            flow: Float[torch.Tensor, "t-1 l 2"] = pos[1:] - pos[:-1]

            # detect before subsampling tracks
            camera_static = self.get_camera_static_tensor(flow, static_camera_flow_mag_threshold, static_camera_fraction_threshold)

            # NOTE: do not reduce number of tracks (yet!)
            pos = pos[:, selected_track_idxs]
            flow = flow[:, selected_track_idxs]

            # A sample is classified as high motion if the maximum flow is above some threshold.
            # Using the max instead of the mean includes samples with low motion background.
            if min_motion_threshold > 0:
                if flow.norm(dim=-1).max() < min_motion_threshold:
                    return {"valid": False}


            visibility = visibility[i_start : i_last+1 : frame_skip]
            visibility = visibility[:, selected_track_idxs]

            return {
                "cond_frame_idx": int(i_start.item()),
                "pos": pos[:-1],  # [t-1, l, 2] in [0, 1]
                "flow": flow,  # [t-1, l, 2] in ~[-1, 1]
                "timeskip": (sample["times"][i_last] - sample["times"][i_start]) / (num_steps - 1),  # in seconds
                "is_camera_static": camera_static,
                "visibility": visibility[:-1],  # [t-1, l]
            }
        except Exception as e:
            print(f"Error: {e=}")
            return {"valid": False}

    def get_camera_static_tensor(self, flow: Float[torch.Tensor, "t n_t 2"], static_camera_flow_mag_threshold: float, static_camera_fraction_threshold: float) -> Bool[torch.Tensor, ""]:
        return (
            flow.norm(dim=-1) < static_camera_flow_mag_threshold
        ).float().mean() > static_camera_fraction_threshold

    def get_camera_static_simple(sample: dict[str, torch.Tensor], static_camera_flow_mag_threshold: float=1, static_camera_fraction_threshold: float=1, track_key="tracks") -> bool:    
        tracks = decode_npy(sample[f"{track_key}.npy"])
        flow = (tracks[1:] - tracks[:-1]) 
        return (
            np.mean(
                np.linalg.norm(flow, axis=-1, ord=2) < static_camera_flow_mag_threshold
            ) > static_camera_fraction_threshold
        )

    @staticmethod
    def is_static_camera_heuristic(sample, *, simple_heuristic=False, static_camera_flow_mag_threshold=1,static_camera_fraction_threshold=1, track_key="tracks"):
        if simple_heuristic:
            return TrackerVideoDataModule.get_camera_static_simple(
                sample, static_camera_flow_mag_threshold, static_camera_fraction_threshold, track_key=track_key
            )
        # need both tracks and visibility logits
        if not (f"{track_key}.npy" in sample and "logits_visible.npy" in sample):
            return False
        tracks = decode_npy(sample[f"{track_key}.npy"])           # (T,N,2) in (y,x), [-1,1] most likely
        vis = 1/(1+np.exp(-decode_npy(sample["logits_visible.npy"]))) > 0.5  # (T,N)
        # convert to (x,y) in [0,1] for geometry
        tracks_xy01 = (tracks + 1)/2
        tracks_xy01 = tracks_xy01[..., [1,0]]
        return (
            np.mean(np.linalg.norm(np.max(tracks_xy01,0)-np.min(tracks_xy01,0), axis=-1) < 0.005) > 0.05
            and check_static_camera_first_last(tracks_xy01, vis, homography_id_thresh=0.02, inlier_ratio_thresh=0.8)[0]
        )

    def _load_frame(self, video, cond_frame: int, image_size: int = 512):
        with io.BytesIO(video) as buf, av.open(buf) as container:
            c_f = 0
            target_frame = None
            for packet in container.demux():
                if not target_frame is None:
                    break
                for frame in packet.decode():
                    if c_f == cond_frame:
                        target_frame = frame.to_ndarray(format="rgb24")
                        break
                    c_f += 1
        if target_frame is None:
            return None
        assert c_f == cond_frame, f"{cond_frame=}, {c_f=}"
        x: Float[torch.Tensor, "c h w"] = augment(
            einops.rearrange(torch.from_numpy(target_frame).float() / 255, "h w c -> 1 c h w"), size=image_size, center_crop=False
        )[
            0
        ].bfloat16()  # [-1, 1]
        return x
    
    def _load_adjusted_frame(
        self,
        video,
        cond_frame: int,
        pos: Float[torch.Tensor, "t l 2"],
        visibility: Float[torch.Tensor, "t l"],
        image_size: int = 512,
        center_crop: bool = False,
    ):
        x = self._load_frame(video, cond_frame, image_size=image_size)
        x, pos = self._center_crop(x, pos, center_crop=center_crop)

        track_in_frame: Bool[torch.Tensor, "t l"] = (
            (pos[:, :, 0] >= 0) & (pos[:, :, 0] <= 1) & (pos[:, :, 1] >= 0) & (pos[:, :, 1] <= 1)
        )
        return x, pos, visibility, track_in_frame
        

    def _center_crop(
        self,
        x: Float[torch.Tensor, "c h w"],
        pos: Float[torch.Tensor, "t l 2"],
        center_crop: bool,
    ):
        if not center_crop:
            return x, pos

        H, W = x.shape[-2], x.shape[-1]
        L = min(H, W)
        starth = (H - L) // 2
        startw = (W - L) // 2

        x = x[..., starth:starth + L, startw:startw + L]

        pos = pos.clone()
        offs = pos.new_tensor([startw, starth])
        scale = pos.new_tensor([W, H])

        pos_px = pos * scale
        pos_px = pos_px - offs
        pos = pos_px / L

        return x, pos

    def _decode(
        self,
        sample: dict[str, bytes],
        track_key: str = "tracks",
        vid_key: str = "video.mp4",
        vis_key: str = "logits_visible",
        txt_key: str | None = None,
        var_thresh: float = 0.0,
        var_scale: float = 1.0,
        num_steps: int = 32,
        num_tracks: int = 16,
        frame_skip: int = 1,
        image_size: int = 512,
        center_crop: bool = False,
        visibility_threshold: float=0.5,
        certainty_threshold: float=0.5,
        min_motion_threshold: float=0.0,
        min_motion_frac: float = 0.0,
        sample_grid: int = 0,
        renorm_tracks: bool = True,
        flip_yx_to_xy: bool = True,
        is_motion_sort: bool = False,
        is_motion_sort_rest: bool = False,
        motion_sortremix_frac: float = 1.0,
        shuffle_tracks: bool = True,
        allow_invisible_track_ends: bool=False,
        allow_out_of_frame_track_ends: bool=False,
        return_video_without_cutting: bool = False,
        return_full_sequence: bool = False,
        static_camera_flow_mag_threshold: float = 0.0007,
        static_camera_fraction_threshold: float = 0.5,
        cond_frame_idx_in_tracks_system: bool = True,
    ) -> dict[str, torch.Tensor | bool]:
        """
        Docstring for _decode
        
        Parameters:
            sample(dict[str, bytes]): The sample from the tar.
            tracks_key(str): The key under which tracks are saved in the samples.
            num_steps(int): length of trajectories to extract.
            num_tracks(int): number of tracks to extract.
            frame_skip(int): distance (in frame indices) between frames (1 = everyframe).
            image_size(int): size to which frames are resized.
            center_crop(bool): whether to center crop the frames.
            visibility_threshold(float): threshold for visibility logits.
            certainty_threshold(float): threshold for certainty values.
            min_motion_threshold(float): minimum threshold for the maximal norm of all extracted flows.
            allow_invisible_track_ends(bool): whether to allow tracks that are invisible at the end frame.
            allow_out_of_frame_track_ends(bool): whether to allow tracks that are out of frame at the end frame.
            return_video_without_cutting(bool): whether to return the video without cutting it to the extracted sample.
            return_full_sequence(bool): whether to return the full sequence of frames instead of just the conditioning frame.
            static_camera_flow_mag_threshold(float): threshold for flow magnitude to consider camera as static.
            static_camera_fraction_threshold(float): fraction threshold for static camera detection.
            cond_frame_idx_in_tracks_system(bool): whether to set the conditioning frame index in the tracks system or the video frame system.
        
        Returns:
            dict[str, torch.Tensor]: the decoded sample with keys:
                `cond_frame`: the conditioning frame (or full video).
                `cond_frame_idx`: the index of the conditioning frame in tracks or video system.
                `tracks`: the extracted tracks.
                `filtering_yield`: the ratio of valid tracks after filtering.
                `is_camera_static`: whether the camera is static according to the heuristic.
                `__key__`: identifier of the sample.
        """
        sample_key = sample["__key__"]
        try:
            for k in [
                vid_key,
                f"{track_key}.npy",
                f"{vis_key}.npy",
            ]:
                assert k in sample.keys(), f"{k=} {sample.keys()=}"

            tracks = torch.from_numpy(decode_npy(sample[f"{track_key}.npy"]))
            if renorm_tracks:
                tracks = (tracks + 1) / 2 # to [0, 1]
            if flip_yx_to_xy:
                tracks = tracks[..., [1, 0]] # (yx) to (xy)

            num_tracks_unfiltered = tracks.shape[1]

            visibility = torch.sigmoid(torch.from_numpy(decode_npy(sample[f"{vis_key}.npy"])).float()).float()

            if "certainty.npy" in sample.keys():
                certainty = torch.from_numpy(decode_npy(sample["certainty.npy"]))
            else:
                certainty = torch.ones_like(visibility)
            if visibility.shape[-1] == 1:
                visibility = visibility.squeeze(dim=-1)
            if certainty.shape[-1] == 1:
                certainty = certainty.squeeze(dim=-1)


            d = {
                "tracks": tracks,  # [t, n_t, 2]
                "visibility": visibility,  # [t, n_t]
                "certainty": certainty,
                "__key__": sample_key,
            }
            
            got_fps = False
            if "meta.json" in sample.keys():
                meta_bytes = sample["meta.json"]
                meta_str = meta_bytes.decode("utf-8")
                meta = json.loads(meta_str)
                if "fps" in meta.keys():
                    fps = float(meta["fps"])
                    got_fps = True
            if "fps" in sample.keys() and not got_fps:
                fps = float(sample["fps"])
                got_fps = True
            if not got_fps:
                fps = 30
            d["times"] = torch.arange(d["tracks"].shape[0]) / fps

            frame_skip = frame_skip if isinstance(frame_skip, int) else random.sample(frame_skip, 1)[0]

            sample_out = self.extract_training_sample(d, frame_skip, num_tracks, num_steps, min_motion_threshold, visibility_threshold, certainty_threshold, static_camera_flow_mag_threshold, static_camera_fraction_threshold, allow_invisible_track_ends)
            if not sample_out.get("valid", True):
                return {"valid": False}
            
            cond_frame_idx = sample_out.get("cond_frame_idx", 0) if not return_video_without_cutting else 0

            x, pos, visibility, track_in_frame = self._load_adjusted_frame(sample[vid_key], cond_frame_idx, sample_out["pos"], sample_out["visibility"], image_size, center_crop,)
            if x is None:
                return {"valid": False}
            if return_full_sequence:
                loaded_frames = [x]
                if return_video_without_cutting:
                    end = int(1e5)
                else:
                    end = cond_frame_idx + num_steps * frame_skip
                for c_frame in range(cond_frame_idx + frame_skip, end, frame_skip):
                    c, _, _, _ = self._load_adjusted_frame(sample[vid_key], c_frame, sample_out["pos"], sample_out["visibility"], image_size, center_crop,)
                    if c == None:
                        break
                    loaded_frames.append(c)
                if len(loaded_frames) == 0:
                    return {"valid": False}
                x = torch.stack(loaded_frames, dim=0)[:-1]  # (t, c, h, w)
                x = einops.rearrange(x, "t c h w -> t h w c")
            else:
                x = einops.rearrange(x, "c h w -> h w c")

            # refilter by updated visibility and reduce to num_tracks
            T, N = visibility.shape
            if var_thresh > 0.0:
                var_pos = torch.var(pos * var_scale, dim=0)
                var_pos = torch.sum(var_pos, dim=-1)
                var_pos = var_pos > var_thresh
            else:
                var_pos = torch.ones((N,), dtype=torch.bool)
            motion_magn = pos[1:] - pos[:1]
            motion_magn = torch.norm(motion_magn, dim=-1)
            motion_magn, _ = motion_magn.max(dim=0)
            valid_tracks_mask = (visibility[0] > visibility_threshold) & var_pos
            valid_tracks_mask = valid_tracks_mask.nonzero()
            valid_tracks_mask = valid_tracks_mask.flatten()
            valid_tracks_mask = valid_tracks_mask[(track_in_frame[0, valid_tracks_mask])]
            if not allow_invisible_track_ends:
                valid_tracks_mask = valid_tracks_mask[(visibility[-1, valid_tracks_mask] > visibility_threshold)]
            if not allow_out_of_frame_track_ends:
                valid_tracks_mask = valid_tracks_mask[(track_in_frame[-1, valid_tracks_mask])]

            final_num_valid_tracks = len(valid_tracks_mask)
            if final_num_valid_tracks < num_tracks:
                return {"valid": False}
            nr_valid = len(valid_tracks_mask)
            selected_track_idxs = valid_tracks_mask[torch.randperm(nr_valid)[:num_tracks]]
            if is_motion_sort or is_motion_sort_rest or motion_sortremix_frac > 0.0:
                valid_idcs = valid_tracks_mask.tolist()
                motion_magn = motion_magn[valid_tracks_mask]
                assert valid_tracks_mask.shape == motion_magn.shape, f"{valid_tracks_mask.shape=} {motion_magn.shape=}"
                motion_magn = motion_magn.tolist()
                assert len(valid_idcs) == len(motion_magn), f"{len(valid_idcs)=} {len(motion_magn)=} {pos.shape=} {valid_tracks_mask.shape=}"
                selected_track_idxs = [idx for _, idx in sorted(zip(motion_magn, valid_idcs))]
                # highest motion first
                selected_track_idxs.reverse()
                if motion_sortremix_frac > 0.0:
                    split_idx = int(num_tracks * motion_sortremix_frac)
                    highest_motion_idxs = selected_track_idxs[:split_idx]
                    rest_motion_idxs = selected_track_idxs[split_idx:]
                    rest_idcs = random.choices(rest_motion_idxs, k=num_tracks - split_idx)
                    selected_track_idxs = highest_motion_idxs + rest_idcs
                    assert len(selected_track_idxs) == num_tracks, f"{len(selected_track_idxs)=} {num_tracks=}"
                    selected_track_idxs = torch.tensor(selected_track_idxs)


                else:
                    highest_motion_idxs = selected_track_idxs[:num_tracks]
                    rest_motion_idxs = selected_track_idxs[num_tracks:]
                    if is_motion_sort_rest:
                        selected_track_idxs = random.sample(rest_motion_idxs, k=num_tracks,) if len(rest_motion_idxs) > num_tracks else selected_track_idxs
                    else:
                        selected_track_idxs = highest_motion_idxs
                    highest_motion_idxs = torch.tensor(highest_motion_idxs)
                    rest_motion_idxs = torch.tensor(rest_motion_idxs)
                    selected_track_idxs = torch.tensor(selected_track_idxs)
            if sample_grid > 0:
                grid_size = sample_grid ** 2
                assert num_tracks % grid_size == 0, f"{sample_grid=} {grid_size=} {num_tracks=}"
                nr_grids = num_tracks // grid_size
                valid_start_pos = pos[0, valid_tracks_mask]
                # [0,1] to [0,G]
                valid_start_pos = valid_start_pos * sample_grid
                valid_start_pos = valid_start_pos.int()
                # `[0,G] per xy` to `x + y * G` (flat grid idcs)
                valid_grid_idcs = valid_start_pos[:, 0] + valid_start_pos[:, 1] * sample_grid
                valid_idcs = torch.arange(nr_valid)
                selected_track_idxs = []
                for _ in range(nr_grids):
                    for grid_idx in range(grid_size):
                        candidates = valid_idcs[valid_grid_idcs == grid_idx].tolist()
                        if len(candidates) == 0:
                            return {"valid": False}
                        candidate = random.sample(candidates, k=1)
                        selected_track_idxs.append(valid_tracks_mask[candidate])
                selected_track_idxs = torch.tensor(selected_track_idxs)
            unfiltered_pos = pos
            pos = unfiltered_pos[:, selected_track_idxs]
            visibility = visibility[:, selected_track_idxs]
            flow = sample_out["flow"][:, selected_track_idxs]
            filter_flow = sample_out["flow"][:, highest_motion_idxs] if (is_motion_sort or is_motion_sort_rest) else flow
            filter_pos = unfiltered_pos[:, highest_motion_idxs] if (is_motion_sort or is_motion_sort_rest) else unfiltered_pos
            if min_motion_threshold > 0:
                if filter_flow.norm(dim=-1).max() < min_motion_threshold:
                    return {"valid": False}
                if min_motion_frac > 0:
                    abs_flow = filter_pos[1:] - filter_pos[:-1]
                    if (abs_flow.norm(dim=-1).max(dim=0)[0] > min_motion_threshold).float().mean() < min_motion_frac:
                        return {"valid": False}

            pos = pos[:, : num_tracks]
            visibility = visibility[:, : num_tracks]
            flow = flow[:, : num_tracks]

            if shuffle_tracks:
                perm = torch.randperm(pos.shape[1])
                pos = pos[:, perm]
                visibility = visibility[:, perm]
                flow = flow[:, perm]
            assert torch.all((pos[0] >= 0) & (pos[0] <= 1)).item(), "pos out of frame at t=0"

            sample_out["tracks"] = pos
            sample_out["visibility"] = visibility
            sample_out["flow"] = flow

            del sample_out['visibility']
            del sample_out["pos"]
            del sample_out['timeskip']
            del sample_out['flow']

            if cond_frame_idx_in_tracks_system:
                sample_out["cond_frame_idx"] = 0

            sample_out = sample_out | {
                "cond_frame": x,
                "filtering_yield": torch.tensor([final_num_valid_tracks / num_tracks_unfiltered], dtype=torch.float32),
            }
            if txt_key is not None:
                assert txt_key in sample.keys(), f"{txt_key=} not in {sample.keys()=}"
                sample_out[txt_key] = sample[txt_key]
            return sample_out
        except Exception as e:
            print(f"Error: {e=}", flush=True)
            return {"valid": False}

    def _filter_valid(self, sample: dict[str, torch.Tensor]) -> bool:
        valid = sample.get("valid", True)
        return valid

    def make_loader(
        self,
        tar_base: str | Path | list[str | Path] | ListConfig,
        shards: str,
        batch_size: int,
        num_workers: int,
        shuffle: int = 0,
        simple_heuristic: bool = False,
        filter_static_camera: bool = False,
        prefetch_factor: int = 2,
        decode_kwargs: dict={},
    ):

        base_dirs = tar_base
        if isinstance(base_dirs, (str, Path)):
            base_dirs = [base_dirs]
        elif isinstance(base_dirs, ListConfig):
            base_dirs = OmegaConf.to_object(base_dirs)
        elif isinstance(base_dirs, list):
            base_dirs = base_dirs
        else:
            raise NotImplementedError(f'Tar base is of type {type(base_dirs)} which is not supported as of now')

        base_dirs = [Path(b).expanduser().resolve() for b in base_dirs]

        if shards is None:
            shard_urls = [
                str(p) for base in base_dirs for p in base.rglob("*.tar")
            ]
        else:
            if isinstance(shards, ListConfig):
                shards = OmegaConf.to_object(shards)
            if isinstance(shards, (list, tuple)):
                patterns = shards
            else:
                patterns = [shards]

            shard_urls = []
            for base in base_dirs:
                for pat in patterns:
                    full_pat = str(base / pat)
                    matches = glob.glob(full_pat)
                    shard_urls.extend(matches)
            if len(shard_urls) == 0:
                raise FileNotFoundError("No shards matched patterns")

        shard_urls = list(set(shard_urls))  # deduplicate
        shard_urls.sort()   # sort

        is_camera_static_heuristic = partial(self.is_static_camera_heuristic, simple_heuristic=simple_heuristic, static_camera_flow_mag_threshold=decode_kwargs.get("static_camera_flow_mag_threshold", 0.0007), static_camera_fraction_threshold=decode_kwargs.get("static_camera_fraction_threshold", 0.5), track_key=decode_kwargs.get("track_key", "tracks"))

        loader_decode = partial(self._decode, **decode_kwargs)

        dataset = wds.DataPipeline(
            wds.SimpleShardList(shard_urls),
            wds.detshuffle() if shuffle != 0 else lambda x: x,
            wds.split_by_node,
            wds.split_by_worker,
            partial(wds.tarfile_samples, handler=wds.warn_and_continue),
            *([wds.shuffle(shuffle)] if shuffle != 0 else []),
            *([wds.select(is_camera_static_heuristic)] if filter_static_camera else []),
            wds.map(loader_decode),
            wds.select(self._filter_valid),
            wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
        )

        has_background_workers = num_workers > 0
        return wds.WebLoader(
            dataset, batch_size=None, num_workers=num_workers, prefetch_factor=prefetch_factor if has_background_workers else None, pin_memory=True, persistent_workers=has_background_workers,
        )

    def train_dataloader(self):
        return self.make_loader(**self.train)

    def val_dataloader(self):
        return self.make_loader(**self.validation)
