"""Training loop for world models."""
from __future__ import annotations

import torch

from data.normalization import NormStats
from training.losses import single_step_loss, multi_step_loss


def train_epoch(model, train_loader, optimizer, norm_stats: NormStats,
                training_mode: str = "single_step", rollout_k: int = 1,
                device: str = "cpu", max_grad_norm: float = 1.0) -> dict:
    """Run one training epoch.

    Args:
        model: WorldModel
        train_loader: DataLoader yielding batches
        optimizer: torch optimizer
        norm_stats: normalization stats
        training_mode: "single_step" or "multi_step"
        rollout_k: horizon for multi-step loss
        device: torch device
        max_grad_norm: gradient clipping threshold

    Returns:
        dict with "train_loss" (mean over batches)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        ns = norm_stats.to(device)

        if training_mode == "single_step":
            loss = single_step_loss(model, batch, ns)
        elif training_mode == "multi_step":
            loss = multi_step_loss(model, batch, ns, k=rollout_k)
        else:
            raise ValueError(
                f"Unsupported training_mode: {training_mode}. "
                f"Available: single_step, multi_step. "
                f"(scheduled_sampling requires GRU — not yet implemented)"
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(model, val_loader, norm_stats: NormStats,
             training_mode: str = "single_step", rollout_k: int = 1,
             device: str = "cpu") -> dict:
    """Run validation. Same loss computation as training, no gradient."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        ns = norm_stats.to(device)

        if training_mode == "single_step":
            loss = single_step_loss(model, batch, ns)
        elif training_mode == "multi_step":
            loss = multi_step_loss(model, batch, ns, k=rollout_k)
        else:
            raise ValueError(
                f"Unsupported training_mode: {training_mode}. "
                f"Available: single_step, multi_step. "
                f"(scheduled_sampling requires GRU — not yet implemented)"
            )

        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}
