"""Episode dataset for world model training.

Loads .npz episodes from a directory. Two modes:
  - "single_step": yields (state, action, delta) tuples for feedforward training
  - "sequence": yields (state_seq, action_seq) windows for multi-step/recurrent training
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    """Load episodes from a directory of .npz files.

    Args:
        data_path: directory containing .npz episode files
        state_dim: number of state dimensions to use (8 = kinematic only, 15 = full)
        mode: "single_step" or "sequence"
        seq_len: sequence length for "sequence" mode
        split: "train", "val", or None (all data)
        val_fraction: fraction of episodes reserved for validation
        seed: seed for deterministic train/val split
    """

    def __init__(
        self,
        data_path: str | Path,
        state_dim: int = 8,
        mode: str = "single_step",
        seq_len: int | None = None,
        split: str | None = None,
        val_fraction: float = 0.1,
        seed: int = 0,
    ):
        self.state_dim = state_dim
        self.action_dim = 2  # Lunar Lander
        self.mode = mode
        self.seq_len = seq_len

        # Load all .npz files sorted for deterministic ordering
        npz_paths = sorted(Path(data_path).glob("*.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"No .npz files in {data_path}")

        # Train/val split by file index
        if split is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(npz_paths))
            n_val = max(1, int(len(npz_paths) * val_fraction))
            val_idx = set(indices[:n_val].tolist())
            if split == "val":
                npz_paths = [p for i, p in enumerate(npz_paths) if i in val_idx]
            elif split == "train":
                npz_paths = [p for i, p in enumerate(npz_paths) if i not in val_idx]

        # Load episodes into memory
        self.states = []    # list of (T+1, state_dim) float32
        self.actions = []   # list of (T, action_dim) float32
        self.deltas = []    # list of (T, state_dim) float32

        for p in npz_paths:
            try:
                d = np.load(p, allow_pickle=False)
            except Exception as e:
                warnings.warn(f"Skipping {p}: {e}")
                continue
            s = d["states"][:, :state_dim].astype(np.float32)
            a = d["actions"].astype(np.float32)
            self.states.append(s)
            self.actions.append(a)
            self.deltas.append((s[1:] - s[:-1]).astype(np.float32))

        self.n_episodes = len(self.states)

        # Build index for __getitem__
        if mode == "single_step":
            # Each item is one (s_t, a_t, delta_t) transition
            self._index = []  # (episode_idx, timestep)
            for i in range(self.n_episodes):
                T = len(self.actions[i])
                for t in range(T):
                    self._index.append((i, t))
        elif mode == "sequence":
            assert seq_len is not None, "seq_len required for sequence mode"
            self._index = []  # (episode_idx, start_timestep)
            for i in range(self.n_episodes):
                T = len(self.actions[i])
                if T < seq_len:
                    continue
                for start in range(T - seq_len + 1):
                    self._index.append((i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        ep_i, t = self._index[idx]
        if self.mode == "single_step":
            return (
                torch.from_numpy(self.states[ep_i][t]),
                torch.from_numpy(self.actions[ep_i][t]),
                torch.from_numpy(self.deltas[ep_i][t]),
            )
        else:  # sequence
            sl = self.seq_len
            return (
                torch.from_numpy(self.states[ep_i][t:t + sl + 1]),
                torch.from_numpy(self.actions[ep_i][t:t + sl]),
            )

    def episode_dicts(self) -> list[dict]:
        """Return episode data as dicts with torch tensors, for norm stat computation."""
        return [
            {
                "states": torch.from_numpy(self.states[i]),
                "deltas": torch.from_numpy(self.deltas[i]),
            }
            for i in range(self.n_episodes)
        ]
