# data/pixel_dataset.py
"""Pixel-space datasets for VAE and dynamics training.

PixelFrameDataset: individual frames for VAE reconstruction training.
PixelEpisodeDataset: sequential (frames, actions) for dynamics training.

Both load .npz files with 'rgb_frames' key. All frames are pre-loaded
into RAM at init (resized + grayscaled), so __getitem__ is a pure
array index with zero I/O. For 5K episodes at 84x84 grayscale,
this uses ~2-3 GB RAM.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _preprocess_frame(frame: np.ndarray, frame_size: int,
                      grayscale: bool) -> np.ndarray:
    """Resize and optionally convert to grayscale. Returns (H, W) or (H, W, 3) uint8."""
    if grayscale and frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(frame, (frame_size, frame_size),
                         interpolation=cv2.INTER_AREA)
    return resized


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert preprocessed frame to (C, H, W) float32 tensor in [0, 1]."""
    if frame.ndim == 2:
        t = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    else:
        t = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
    return t


def _split_episodes(npz_files: list[Path], split: str | None,
                    val_fraction: float, seed: int) -> list[Path]:
    """Split episode files into train/val by episode."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(npz_files))
    n_val = max(1, int(len(npz_files) * val_fraction))
    if split == "val":
        selected = sorted(indices[:n_val])
    elif split == "train":
        selected = sorted(indices[n_val:])
    else:
        selected = list(range(len(npz_files)))
    return [npz_files[i] for i in selected]


def _load_and_preprocess_all_frames(
    episode_files: list[Path], frame_size: int, grayscale: bool
) -> list[np.ndarray]:
    """Load all episodes, preprocess frames, return list of (T+1, H, W) or (T+1, H, W, 3) arrays."""
    all_episodes = []
    for path in tqdm(episode_files, desc="Loading frames", unit="ep"):
        try:
            with np.load(str(path)) as data:
                raw_frames = data["rgb_frames"]
        except Exception:
            continue
        # Preprocess each frame in the episode
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(len(raw_frames))
        ])  # (T+1, H, W) or (T+1, H, W, 3)
        all_episodes.append(processed)
    return all_episodes


class PixelFrameDataset(Dataset):
    """Individual frames for VAE training.

    Pre-loads all frames into RAM at init. __getitem__ is a pure
    array index — zero I/O during training.

    Args:
        data_path: directory with .npz episode files
        frame_size: target resolution (square)
        grayscale: convert to grayscale
        split: "train", "val", or None
        val_fraction: fraction for validation
        seed: RNG seed for split
    """

    def __init__(self, data_path: str | Path, frame_size: int = 84,
                 grayscale: bool = True, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0):
        self.frame_size = frame_size
        self.grayscale = grayscale

        data_path = Path(data_path)
        npz_files = sorted(data_path.glob("**/*.npz"))
        npz_files = [f for f in npz_files if "prepared" not in f.name]
        episode_files = _split_episodes(npz_files, split, val_fraction, seed)

        # Pre-load all frames into one flat array
        episodes = _load_and_preprocess_all_frames(
            episode_files, frame_size, grayscale)

        if episodes:
            # Concatenate all frames into single array
            self._frames = np.concatenate(episodes, axis=0)  # (N, H, W) or (N, H, W, 3)
        else:
            if grayscale:
                self._frames = np.zeros((0, frame_size, frame_size), dtype=np.uint8)
            else:
                self._frames = np.zeros((0, frame_size, frame_size, 3), dtype=np.uint8)

        n_total = self._frames.shape[0]
        mb = self._frames.nbytes / 1024 / 1024
        print(f"PixelFrameDataset: {n_total} frames from {len(episodes)} episodes "
              f"({mb:.0f} MB in RAM)")

    def __len__(self) -> int:
        return self._frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _frame_to_tensor(self._frames[idx])


class PixelEpisodeDataset(Dataset):
    """Sequential frame+action chunks for dynamics training.

    Pre-loads all frames and actions into RAM. __getitem__ builds
    stacked-frame windows from the pre-loaded arrays — zero I/O.

    Args:
        data_path: directory with .npz episode files
        frame_size: target resolution
        grayscale: convert to grayscale
        seq_len: number of timesteps per window
        frame_stack: number of frames to stack (1=no stacking)
        split, val_fraction, seed: train/val splitting
    """

    def __init__(self, data_path: str | Path, frame_size: int = 84,
                 grayscale: bool = True, seq_len: int = 20,
                 frame_stack: int = 1, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0):
        self.frame_size = frame_size
        self.grayscale = grayscale
        self.seq_len = seq_len
        self.frame_stack = frame_stack

        data_path = Path(data_path)
        npz_files = sorted(data_path.glob("**/*.npz"))
        npz_files = [f for f in npz_files if "prepared" not in f.name]
        episode_files = _split_episodes(npz_files, split, val_fraction, seed)

        min_frames = seq_len + frame_stack - 1 + 1

        # Pre-load frames and actions per episode, build window index
        self._episode_frames = []  # list of (T+1, H, W) uint8 arrays
        self._episode_actions = []  # list of (T, action_dim) float32 arrays
        self._window_index = []  # (episode_idx, start_frame)

        for path in tqdm(episode_files, desc="Loading episodes", unit="ep"):
            try:
                with np.load(str(path)) as data:
                    raw_frames = data["rgb_frames"]
                    actions = data["actions"].astype(np.float32)
            except Exception:
                continue

            n_frames = raw_frames.shape[0]
            n_actions = actions.shape[0]
            usable = min(n_frames, n_actions + 1)
            if usable < min_frames:
                continue

            # Preprocess all frames for this episode
            processed = np.stack([
                _preprocess_frame(raw_frames[i], frame_size, grayscale)
                for i in range(usable)
            ])  # (usable, H, W)

            ep_idx = len(self._episode_frames)
            self._episode_frames.append(processed)
            self._episode_actions.append(actions[:usable - 1])

            # Build window indices
            max_start = usable - seq_len - frame_stack + 1
            for s in range(frame_stack - 1, frame_stack - 1 + max_start):
                self._window_index.append((ep_idx, s))

        total_frames = sum(f.shape[0] for f in self._episode_frames)
        mb = sum(f.nbytes for f in self._episode_frames) / 1024 / 1024
        mb += sum(a.nbytes for a in self._episode_actions) / 1024 / 1024
        print(f"PixelEpisodeDataset: {len(self._window_index)} windows from "
              f"{len(self._episode_frames)} episodes, {total_frames} frames "
              f"({mb:.0f} MB in RAM)")

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, start = self._window_index[idx]
        frames = self._episode_frames[ep_idx]
        actions = self._episode_actions[ep_idx]

        frame_start = start - (self.frame_stack - 1)
        frame_end = start + self.seq_len
        raw_chunk = frames[frame_start:frame_end]  # already preprocessed uint8
        act_chunk = actions[start:start + self.seq_len]

        # Convert frames to tensors
        processed = [_frame_to_tensor(raw_chunk[i]) for i in range(len(raw_chunk))]

        # Build stacked frames
        stacked_frames = []
        for t in range(self.seq_len):
            stack = torch.cat(processed[t:t + self.frame_stack], dim=0)
            stacked_frames.append(stack)

        frames_tensor = torch.stack(stacked_frames, dim=0)
        actions_tensor = torch.from_numpy(act_chunk.copy()).float()

        return frames_tensor, actions_tensor
