"""Training loop for world models."""
from __future__ import annotations

import torch

from data.normalization import NormStats
from training.losses import single_step_loss, multi_step_loss, scheduled_sampling_loss, elbo_loss


def train_epoch(model, train_loader, optimizer, norm_stats: NormStats,
                training_mode: str = "single_step", rollout_k: int = 1,
                device: str = "cpu", max_grad_norm: float = 1.0,
                sampling_prob: float = 0.0, kl_weight: float = 1.0,
                ctx=None, callbacks=None) -> dict:
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
        sampling_prob: probability of using model predictions (scheduled sampling)
        kl_weight: weight for KL divergence term (ELBO)
        ctx: optional CallbackContext (updated in-place with global_step)
        callbacks: optional list of TrainCallback (dispatched per step)

    Returns:
        dict with "train_loss" (mean over batches), "stop_requested" (bool)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    stop_requested = False

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        ns = norm_stats.to(device)

        if training_mode == "single_step":
            loss = single_step_loss(model, batch, ns)
        elif training_mode == "multi_step":
            loss = multi_step_loss(model, batch, ns, k=rollout_k)
        elif training_mode == "scheduled_sampling":
            loss = scheduled_sampling_loss(model, batch, ns, k=rollout_k,
                                           sampling_prob=sampling_prob)
        elif training_mode == "elbo":
            loss = elbo_loss(model, batch, ns, k=rollout_k, kl_weight=kl_weight)
        else:
            raise ValueError(
                f"Unsupported training_mode: {training_mode}. "
                f"Available: single_step, multi_step, scheduled_sampling, elbo"
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Dispatch on_step callbacks AFTER backward+clip, BEFORE optimizer.step()
        if ctx is not None and callbacks:
            ctx.global_step += 1
            ctx.extras["train_loss_step"] = loss.item()
            if ctx.writer:
                ctx.writer.add_scalar("train/loss", loss.item(), ctx.global_step)
            for cb in callbacks:
                if cb.on_step(ctx) is False:
                    stop_requested = True
                    break

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if stop_requested:
            break

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "stop_requested": stop_requested,
    }


@torch.no_grad()
def validate(model, val_loader, norm_stats: NormStats,
             training_mode: str = "single_step", rollout_k: int = 1,
             device: str = "cpu", sampling_prob: float = 0.0,
             kl_weight: float = 1.0) -> dict:
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
        elif training_mode == "scheduled_sampling":
            loss = scheduled_sampling_loss(model, batch, ns, k=rollout_k,
                                           sampling_prob=sampling_prob)
        elif training_mode == "elbo":
            loss = elbo_loss(model, batch, ns, k=rollout_k, kl_weight=kl_weight)
        else:
            raise ValueError(
                f"Unsupported training_mode: {training_mode}. "
                f"Available: single_step, multi_step, scheduled_sampling, elbo"
            )

        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}
