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

import os
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


def _load_one_episode(args: tuple) -> np.ndarray | None:
    """Load and preprocess a single episode's frames. For multiprocessing.Pool."""
    path, frame_size, grayscale = args
    try:
        with np.load(str(path)) as data:
            raw_frames = data["rgb_frames"]
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(len(raw_frames))
        ])
        return processed
    except Exception:
        return None


def _load_one_episode_states(args: tuple) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load frames + states from a single episode. For multiprocessing.Pool."""
    path, frame_size, grayscale, state_dim = args
    try:
        with np.load(str(path)) as data:
            raw_frames = data["rgb_frames"]
            # States are (T+1, full_state_dim), slice to state_dim
            states = data["states"][:, :state_dim].astype(np.float32) if "states" in data else None
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(len(raw_frames))
        ])
        return processed, states
    except Exception:
        return None, None


def _load_one_episode_with_actions(args: tuple) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load and preprocess a single episode's frames + actions. For multiprocessing.Pool."""
    path, frame_size, grayscale = args
    try:
        with np.load(str(path)) as data:
            raw_frames = data["rgb_frames"]
            actions = data["actions"].astype(np.float32)
        n_frames = raw_frames.shape[0]
        n_actions = actions.shape[0]
        usable = min(n_frames, n_actions + 1)
        processed = np.stack([
            _preprocess_frame(raw_frames[i], frame_size, grayscale)
            for i in range(usable)
        ])
        return processed, actions[:usable - 1]
    except Exception:
        return None, None


def _load_and_preprocess_all_frames(
    episode_files: list[Path], frame_size: int, grayscale: bool,
    n_workers: int = 8,
) -> list[np.ndarray]:
    """Load all episodes in parallel, preprocess frames.

    Uses multiprocessing.Pool to parallelize npz decompression + resize.
    Returns list of (T+1, H, W) or (T+1, H, W, 3) uint8 arrays.
    """
    from multiprocessing import Pool

    args = [(path, frame_size, grayscale) for path in episode_files]

    all_episodes = []
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap(_load_one_episode, args),
            total=len(args), desc="Loading frames", unit="ep",
        ):
            if result is not None:
                all_episodes.append(result)
    return all_episodes


class PixelFrameDataset(Dataset):
    """Individual frames for VAE training.

    Pre-loads all frames into RAM at init. __getitem__ is a pure
    array index — zero I/O during training.

    If cache_path is provided, saves/loads preprocessed frames as a
    single .npy file for instant loading on subsequent runs (~2s vs ~10min).
    Cache is per-split: include the split name in cache_path.

    When state_dim > 0, also loads kinematic states and returns
    (frame, state) tuples instead of just frames.

    Args:
        data_path: directory with .npz episode files
        frame_size: target resolution (square)
        grayscale: convert to grayscale
        split: "train", "val", or None
        val_fraction: fraction for validation
        seed: RNG seed for split
        cache_path: optional path to save/load preprocessed frames (.npy)
        state_dim: if > 0, load and return states (first N dims from .npz states)
    """

    def __init__(self, data_path: str | Path | list[str | Path],
                 frame_size: int = 84,
                 grayscale: bool = True, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0,
                 n_workers: int = 8, cache_path: str | Path | None = None,
                 state_dim: int = 0):
        self.frame_size = frame_size
        self.grayscale = grayscale
        self.state_dim = state_dim
        self._states = None  # (N, state_dim) float32 or None

        # Try loading from cache first.
        # Cache is a directory with frames.npy and optionally states.npy.
        # Uses mmap_mode='r' — memory-mapped, no RAM allocation. OS pages
        # data from disk on demand. DataLoader workers handle parallelism.
        if cache_path is not None and Path(cache_path).is_dir():
            frames_file = Path(cache_path) / "frames.npy"
            if frames_file.exists():
                print(f"Loading from cache (mmap): {cache_path} ...")
                self._frames = np.load(str(frames_file), mmap_mode='r')
                states_file = Path(cache_path) / "states.npy"
                if state_dim > 0:
                    if not states_file.exists():
                        raise ValueError(
                            f"Cache at {cache_path} has no states.npy, but state_dim={state_dim}. "
                            f"Delete the cache dir and rerun."
                        )
                    self._states = np.load(str(states_file), mmap_mode='r')
                n = self._frames.shape[0]
                disk_mb = os.path.getsize(str(frames_file)) / 1024 / 1024
                state_str = f" + {self._states.shape[1]}D states" if self._states is not None else ""
                print(f"PixelFrameDataset: {n} frames{state_str} ({disk_mb:.0f} MB on disk, mmap)")
                return

        # Normalize to list of paths
        if isinstance(data_path, (str, Path)):
            data_path = [data_path]
        npz_files = []
        for dp in data_path:
            for root, _dirs, files in os.walk(str(Path(dp)), followlinks=True):
                for f in files:
                    if f.endswith(".npz"):
                        npz_files.append(Path(root) / f)
        npz_files.sort()
        npz_files = [f for f in npz_files if "prepared" not in f.name]
        episode_files = _split_episodes(npz_files, split, val_fraction, seed)

        if state_dim > 0:
            # Load frames + states
            load_args = [(path, frame_size, grayscale, state_dim) for path in episode_files]
            all_frames = []
            all_states = []
            if n_workers > 1:
                from multiprocessing import Pool
                with Pool(n_workers) as pool:
                    iterator = pool.imap(_load_one_episode_states, load_args)
                    for frames, states in tqdm(iterator, total=len(load_args),
                                               desc="Loading frames+states", unit="ep"):
                        if frames is not None:
                            all_frames.append(frames)
                            if states is not None:
                                all_states.append(states)
            else:
                for a in tqdm(load_args, desc="Loading frames+states", unit="ep"):
                    frames, states = _load_one_episode_states(a)
                    if frames is not None:
                        all_frames.append(frames)
                        if states is not None:
                            all_states.append(states)

            if all_frames:
                self._frames = np.concatenate(all_frames, axis=0)
                if all_states:
                    self._states = np.concatenate(all_states, axis=0).astype(np.float32)
            else:
                self._frames = np.zeros((0, frame_size, frame_size), dtype=np.uint8)
        else:
            # Frames only (original path)
            episodes = _load_and_preprocess_all_frames(
                episode_files, frame_size, grayscale, n_workers=n_workers)
            if episodes:
                self._frames = np.concatenate(episodes, axis=0)
            else:
                if grayscale:
                    self._frames = np.zeros((0, frame_size, frame_size), dtype=np.uint8)
                else:
                    self._frames = np.zeros((0, frame_size, frame_size, 3), dtype=np.uint8)

        n_total = self._frames.shape[0]
        mb = self._frames.nbytes / 1024 / 1024
        if self._states is not None:
            mb += self._states.nbytes / 1024 / 1024
        print(f"PixelFrameDataset: {n_total} frames"
              f"{f' + {self.state_dim}D states' if self._states is not None else ''}"
              f" from {len(episode_files)} episodes ({mb:.0f} MB in RAM)")

        # Save cache, then free RAM and reload as mmap.
        # np.save writes directly without copying. After saving, delete
        # the in-memory arrays and reload as memory-mapped — OS pages
        # from disk on demand, near-zero RAM.
        if cache_path is not None and n_total > 0:
            cache_dir = Path(cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            frames_file = cache_dir / "frames.npy"
            np.save(str(frames_file), self._frames)
            states_file = cache_dir / "states.npy"
            if self._states is not None:
                np.save(str(states_file), self._states)
            print(f"Saved cache to {cache_dir}")

            # Free RAM and reload as mmap
            del self._frames
            self._frames = np.load(str(frames_file), mmap_mode='r')
            if self._states is not None:
                del self._states
                self._states = np.load(str(states_file), mmap_mode='r')
            print(f"Reloaded as mmap (RAM freed)")

    def __len__(self) -> int:
        return self._frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        frame = _frame_to_tensor(np.array(self._frames[idx]))  # copy from mmap
        if self._states is not None:
            state = torch.from_numpy(np.array(self._states[idx])).float()
            return frame, state
        return frame


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

    def __init__(self, data_path: str | Path | list[str | Path],
                 frame_size: int = 84,
                 grayscale: bool = True, seq_len: int = 20,
                 frame_stack: int = 1, split: str | None = None,
                 val_fraction: float = 0.1, seed: int = 0,
                 n_workers: int = 8, cache_path: str | Path | None = None):
        self.frame_size = frame_size
        self.grayscale = grayscale
        self.seq_len = seq_len
        self.frame_stack = frame_stack

        min_frames = seq_len + frame_stack - 1 + 1

        # Pre-load frames and actions per episode, build window index
        self._episode_frames = []  # list of (T+1, H, W) uint8 arrays
        self._episode_actions = []  # list of (T, action_dim) float32 arrays
        self._window_index = []  # (episode_idx, start_frame)

        # Try cache first
        if cache_path is not None and Path(cache_path).exists():
            print(f"Loading cached episodes from {cache_path} ...")
            cached = np.load(str(cache_path), allow_pickle=True)
            all_frames = cached["all_frames"]  # object array of per-episode frames
            all_actions = cached["all_actions"]  # object array of per-episode actions
            for i in range(len(all_frames)):
                frames = all_frames[i]
                if frames.shape[0] < min_frames:
                    continue
                ep_idx = len(self._episode_frames)
                self._episode_frames.append(frames)
                self._episode_actions.append(all_actions[i])
                max_start = frames.shape[0] - seq_len - frame_stack + 1
                for s in range(frame_stack - 1, frame_stack - 1 + max_start):
                    self._window_index.append((ep_idx, s))
        else:
            if isinstance(data_path, (str, Path)):
                data_path = [data_path]
            npz_files = []
            for dp in data_path:
                for root, dirs, files in os.walk(str(Path(dp)), followlinks=True):
                    for f in files:
                        if f.endswith(".npz"):
                            npz_files.append(Path(root) / f)
            npz_files.sort()
            npz_files = [f for f in npz_files if "prepared" not in f.name]
            episode_files = _split_episodes(npz_files, split, val_fraction, seed)

            from multiprocessing import Pool

            args = [(path, frame_size, grayscale) for path in episode_files]
            with Pool(n_workers) as pool:
                for frames, actions in tqdm(
                    pool.imap(_load_one_episode_with_actions, args),
                    total=len(args), desc="Loading episodes", unit="ep",
                ):
                    if frames is None:
                        continue
                    if frames.shape[0] < min_frames:
                        continue

                    ep_idx = len(self._episode_frames)
                    self._episode_frames.append(frames)
                    self._episode_actions.append(actions)

                    max_start = frames.shape[0] - seq_len - frame_stack + 1
                    for s in range(frame_stack - 1, frame_stack - 1 + max_start):
                        self._window_index.append((ep_idx, s))

            # Save cache
            if cache_path is not None and self._episode_frames:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                # Use object arrays to store variable-length episodes
                np.savez(
                    str(cache_path),
                    all_frames=np.array(self._episode_frames, dtype=object),
                    all_actions=np.array(self._episode_actions, dtype=object),
                )
                print(f"Saved episode cache to {cache_path}")

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
