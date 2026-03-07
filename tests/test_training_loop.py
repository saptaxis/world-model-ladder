import torch
from torch.utils.data import DataLoader

from data.loader import EpisodeDataset
from data.normalization import NormStats, compute_norm_stats
from models.mlp import MLPModel
from training.loop import train_epoch, validate


def _make_norm_stats():
    return NormStats(
        state_mean=torch.zeros(8), state_std=torch.ones(8),
        delta_mean=torch.zeros(8), delta_std=torch.ones(8),
    )


def test_train_epoch_single_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ns = _make_norm_stats()
    metrics = train_epoch(model, loader, optimizer, ns,
                          training_mode="single_step", rollout_k=1)
    assert "train_loss" in metrics
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 100  # sanity — not exploded


def test_train_epoch_multi_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ns = _make_norm_stats()
    metrics = train_epoch(model, loader, optimizer, ns,
                          training_mode="multi_step", rollout_k=5)
    assert "train_loss" in metrics
    assert metrics["train_loss"] > 0


def test_validate(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=16)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    ns = _make_norm_stats()
    metrics = validate(model, loader, ns, training_mode="single_step", rollout_k=1)
    assert "val_loss" in metrics
    assert metrics["val_loss"] > 0


from models.gru import GRUModel


def test_train_epoch_gru_multi_step(episode_dir):
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="sequence", seq_len=10)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_stats = compute_norm_stats(ds.episode_dicts())
    metrics = train_epoch(model, loader, optimizer, norm_stats,
                          training_mode="multi_step", rollout_k=5)
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 1000
