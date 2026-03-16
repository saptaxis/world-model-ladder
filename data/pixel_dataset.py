# data/pixel_dataset.py
"""Pixel-space datasets for VAE and dynamics training.

PixelFrameDataset: individual frames for VAE reconstruction training.
PixelEpisodeDataset: sequential (frames, actions) for dynamics training.

Both load .npz files with 'rgb_frames' key. Preprocessing (resize,
grayscale, normalize) happens on the fly.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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


class PixelFrameDataset(Dataset):
    """Individual frames for VAE training."""

    def __init__(self, data_path: str | Path, frame_size: int = 84,
                 grayscale: bool = True, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0):
        self.frame_size = frame_size
        self.grayscale = grayscale

        data_path = Path(data_path)
        npz_files = sorted(data_path.glob("**/*.npz"))
        npz_files = [f for f in npz_files if "prepared" not in f.name]

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(npz_files))
        n_val = max(1, int(len(npz_files) * val_fraction))
        if split == "val":
            episode_indices = sorted(indices[:n_val])
        elif split == "train":
            episode_indices = sorted(indices[n_val:])
        else:
            episode_indices = list(range(len(npz_files)))

        self._frame_index = []
        for ei in episode_indices:
            path = npz_files[ei]
            try:
                with np.load(str(path)) as data:
                    n_frames = data["rgb_frames"].shape[0]
                self._frame_index.extend([(path, fi) for fi in range(n_frames)])
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, frame_idx = self._frame_index[idx]
        with np.load(str(path)) as data:
            frame = data["rgb_frames"][frame_idx]
        frame = _preprocess_frame(frame, self.frame_size, self.grayscale)
        return _frame_to_tensor(frame)


class PixelEpisodeDataset(Dataset):
    """Sequential frame+action chunks for dynamics training."""

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

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(npz_files))
        n_val = max(1, int(len(npz_files) * val_fraction))
        if split == "val":
            episode_indices = sorted(indices[:n_val])
        elif split == "train":
            episode_indices = sorted(indices[n_val:])
        else:
            episode_indices = list(range(len(npz_files)))

        min_frames = seq_len + frame_stack - 1 + 1
        self._window_index = []
        for ei in episode_indices:
            path = npz_files[ei]
            try:
                with np.load(str(path)) as data:
                    n_frames = data["rgb_frames"].shape[0]
                    n_actions = data["actions"].shape[0]
            except Exception:
                continue
            usable = min(n_frames, n_actions + 1)
            if usable < min_frames:
                continue
            max_start = usable - seq_len - frame_stack + 1
            for s in range(frame_stack - 1, frame_stack - 1 + max_start):
                self._window_index.append((path, s))

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, start = self._window_index[idx]
        with np.load(str(path)) as data:
            frame_start = start - (self.frame_stack - 1)
            frame_end = start + self.seq_len
            raw_frames = data["rgb_frames"][frame_start:frame_end]
            actions = data["actions"][start:start + self.seq_len]

        processed = []
        for f in raw_frames:
            pf = _preprocess_frame(f, self.frame_size, self.grayscale)
            processed.append(_frame_to_tensor(pf))

        stacked_frames = []
        for t in range(self.seq_len):
            stack = torch.cat(processed[t:t + self.frame_stack], dim=0)
            stacked_frames.append(stack)

        frames_tensor = torch.stack(stacked_frames, dim=0)
        actions_tensor = torch.from_numpy(actions.copy()).float()

        return frames_tensor, actions_tensor
