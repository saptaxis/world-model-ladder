"""Loss functions for world model training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from data.normalization import NormStats, normalize, denormalize


def single_step_loss(model, batch, norm_stats: NormStats) -> torch.Tensor:
    """MSE on single-step delta prediction.

    Model contract: receives state-normalized obs, outputs delta-normalized delta.

    Args:
        batch: (obs, actions, true_deltas) each [batch, dim]
        norm_stats: normalization statistics
    """
    obs, actions, true_deltas = batch
    obs_n = normalize(obs, norm_stats.state_mean, norm_stats.state_std)
    pred_deltas_n, _ = model.step(obs_n, actions)
    true_deltas_n = normalize(true_deltas, norm_stats.delta_mean, norm_stats.delta_std)
    return F.mse_loss(pred_deltas_n, true_deltas_n)


def multi_step_loss(model, batch, norm_stats: NormStats, k: int) -> torch.Tensor:
    """MSE on k-step autoregressive rollout.

    Rollout in RAW space: at each step, normalize state for model input,
    get delta-normalized prediction, denormalize delta back to raw, accumulate.
    This avoids conflating state normalization and delta normalization.

    Does NOT use rollout_open_loop (which is normalization-unaware).
    Backpropagates through the entire k-step chain.

    Args:
        batch: (state_seq, action_seq) where state_seq is [batch, T+1, state_dim]
               and action_seq is [batch, T, action_dim]
        norm_stats: normalization statistics
        k: number of rollout steps
    """
    state_seq, action_seq = batch
    s0 = state_seq[:, 0]
    actions_k = action_seq[:, :k]
    true_states_k = state_seq[:, :k + 1]

    # Rollout in raw space with normalize/denormalize at each step
    pred_deltas_raw = []
    s = s0
    model_state = None
    for t in range(k):
        s_n = normalize(s, norm_stats.state_mean, norm_stats.state_std)
        delta_n, model_state = model.step(s_n, actions_k[:, t], model_state)
        delta_raw = denormalize(delta_n, norm_stats.delta_mean, norm_stats.delta_std)
        s = s + delta_raw
        pred_deltas_raw.append(delta_raw)
    pred_deltas_raw = torch.stack(pred_deltas_raw, dim=1)

    # Loss in delta-normalized space
    true_deltas = true_states_k[:, 1:] - true_states_k[:, :-1]
    true_deltas_n = normalize(true_deltas, norm_stats.delta_mean, norm_stats.delta_std)
    pred_deltas_n = normalize(pred_deltas_raw, norm_stats.delta_mean, norm_stats.delta_std)

    return F.mse_loss(pred_deltas_n, true_deltas_n)
