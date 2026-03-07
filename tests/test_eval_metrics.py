import torch
import numpy as np
from torch.utils.data import DataLoader

from data.loader import EpisodeDataset
from data.normalization import NormStats
from models.mlp import MLPModel
from evaluation.metrics.core import (
    per_dim_mse,
    horizon_error_curve,
    divergence_exponent,
    horizon_to_failure,
)


def _make_norm_stats():
    return NormStats(
        state_mean=torch.zeros(8), state_std=torch.ones(8),
        delta_mean=torch.zeros(8), delta_std=torch.ones(8),
    )


def test_per_dim_mse(episode_dir):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    ds = EpisodeDataset(episode_dir, state_dim=8, mode="single_step")
    loader = DataLoader(ds, batch_size=32)
    result = per_dim_mse(model, loader, _make_norm_stats())
    assert result.shape == (8,)
    assert (result >= 0).all()


def test_horizon_error_curve(episode_dir):
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    ds = EpisodeDataset(episode_dir, state_dim=8)
    ns = _make_norm_stats()
    horizons = [1, 5, 10]
    curves = horizon_error_curve(model, ds, ns, horizons=horizons)
    assert set(curves.keys()) == set(horizons)
    for h in horizons:
        assert curves[h].shape == (8,)  # per-dim MSE at each horizon
        assert (curves[h] >= 0).all()


def test_divergence_exponent():
    # Exponential growth: error(h) = 0.1 * e^(0.05 * h)
    horizons = [1, 5, 10, 20, 50]
    errors = {h: 0.1 * np.exp(0.05 * h) for h in horizons}
    lam = divergence_exponent(errors)
    assert abs(lam - 0.05) < 0.01


def test_horizon_to_failure():
    horizons = [1, 5, 10, 20, 50]
    errors = {1: 0.01, 5: 0.05, 10: 0.2, 20: 0.8, 50: 5.0}
    htf = horizon_to_failure(errors, threshold=0.5)
    assert htf == 10  # last horizon where error < threshold


def test_horizon_to_failure_never_fails():
    errors = {1: 0.01, 5: 0.02, 10: 0.03}
    htf = horizon_to_failure(errors, threshold=1.0)
    assert htf == 10  # never exceeded
