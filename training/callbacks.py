"""Training callback system for world models.

Provides a CallbackContext (shared state across callbacks) and a TrainCallback
base class with hooks dispatched at each gradient step, epoch end, and
training start/end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from data.normalization import NormStats
from evaluation.metrics.core import per_dim_mse, horizon_error_curve
from training.loop import validate
from utils.checkpoint import save_checkpoint


@dataclass
class CallbackContext:
    """Shared state passed to all callbacks."""
    model: nn.Module
    optimizer: Optimizer
    writer: Any  # SummaryWriter or None
    global_step: int
    epoch: int
    run_dir: str
    device: str
    extras: dict = field(default_factory=dict)


class TrainCallback:
    """Base class for training callbacks."""

    def on_train_start(self, ctx: CallbackContext) -> None:
        pass

    def on_step(self, ctx: CallbackContext) -> bool:
        return True

    def on_epoch_end(self, ctx: CallbackContext) -> bool:
        return True

    def on_train_end(self, ctx: CallbackContext) -> None:
        pass


class ValidationCallback(TrainCallback):
    """Runs validation every N steps with early stopping and best-checkpoint saving."""

    def __init__(self, val_loader, norm_stats: NormStats,
                 training_mode: str = "single_step",
                 every_n_steps: int = 500, patience: int = 10,
                 checkpoint_dir: str | None = None,
                 rollout_k: int = 1, sampling_prob: float = 0.0,
                 kl_weight: float = 1.0):
        self.val_loader = val_loader
        self.norm_stats = norm_stats
        self.training_mode = training_mode
        self.every_n_steps = every_n_steps
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.rollout_k = rollout_k
        self.sampling_prob = sampling_prob
        self.kl_weight = kl_weight
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.last_val_step = -1

    def on_train_start(self, ctx):
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_step(self, ctx):
        if ctx.global_step == 0:
            return True
        if ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.global_step == self.last_val_step:
            return True
        self.last_val_step = ctx.global_step

        val_metrics = validate(
            ctx.model, self.val_loader, self.norm_stats,
            training_mode=self.training_mode, rollout_k=self.rollout_k,
            device=ctx.device, sampling_prob=self.sampling_prob,
            kl_weight=self.kl_weight,
        )
        val_loss = val_metrics["val_loss"]
        ctx.extras["val_loss"] = val_loss

        if ctx.writer:
            ctx.writer.add_scalar("val/loss", val_loss, ctx.global_step)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.checkpoint_dir and "config" in ctx.extras:
                save_checkpoint(
                    Path(self.checkpoint_dir) / "best.pt",
                    ctx.model, ctx.optimizer, self.norm_stats,
                    ctx.extras["config"], ctx.epoch,
                    {"val_loss": val_loss, "step": ctx.global_step},
                )
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"Early stopping at step {ctx.global_step} "
                  f"(patience {self.patience} exhausted, best val_loss={self.best_val_loss:.6f})")
            return False

        return True


class CheckpointCallback(TrainCallback):
    """Saves periodic checkpoints and a latest checkpoint at each epoch end."""

    def __init__(self, checkpoint_dir: str, every_n_steps: int = 2000):
        self.checkpoint_dir = checkpoint_dir
        self.every_n_steps = every_n_steps

    def on_train_start(self, ctx):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _save(self, ctx, filename):
        path = Path(self.checkpoint_dir) / filename
        metrics = {}
        if "val_loss" in ctx.extras:
            metrics["val_loss"] = ctx.extras["val_loss"]
        norm_stats = ctx.extras.get("norm_stats")
        config = ctx.extras.get("config")
        if norm_stats is None or config is None:
            # Minimal save without full checkpoint infrastructure
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": ctx.model.state_dict(),
                "optimizer_state_dict": ctx.optimizer.state_dict(),
                "epoch": ctx.epoch,
                "metrics": metrics,
            }, path)
        else:
            save_checkpoint(
                path, ctx.model, ctx.optimizer,
                norm_stats, config, ctx.epoch, metrics,
            )

    def on_step(self, ctx):
        if ctx.global_step > 0 and ctx.global_step % self.every_n_steps == 0:
            self._save(ctx, f"step_{ctx.global_step:05d}.pt")
        return True

    def on_epoch_end(self, ctx):
        self._save(ctx, "latest.pt")
        return True


class PerDimLossCallback(TrainCallback):
    """Logs per-dimension MSE at regular intervals."""

    def __init__(self, val_loader, norm_stats: NormStats,
                 every_n_steps: int = 500,
                 dim_names: list[str] | None = None):
        self.val_loader = val_loader
        self.norm_stats = norm_stats
        self.every_n_steps = every_n_steps
        self.dim_names = dim_names

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        pdm = per_dim_mse(ctx.model, self.val_loader, self.norm_stats, device=ctx.device)
        ctx.extras["per_dim_mse"] = pdm.tolist()
        names = self.dim_names or [f"dim_{i}" for i in range(len(pdm))]
        if ctx.writer:
            for i, name in enumerate(names[:len(pdm)]):
                ctx.writer.add_scalar(f"loss_dim/{name}", pdm[i].item(), ctx.global_step)
        return True


class RolloutMetricsCallback(TrainCallback):
    """Computes horizon error curves at regular intervals."""

    def __init__(self, dataset, norm_stats: NormStats,
                 horizons: list[int] | None = None,
                 every_n_steps: int = 2000, n_rollouts: int = 10):
        self.dataset = dataset
        self.norm_stats = norm_stats
        self.horizons = horizons or [1, 5, 10, 20, 50]
        self.every_n_steps = every_n_steps
        self.n_rollouts = n_rollouts

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        curves = horizon_error_curve(
            ctx.model, self.dataset, self.norm_stats,
            horizons=self.horizons, n_rollouts=self.n_rollouts,
            device=ctx.device,
        )
        ctx.extras["horizon_errors"] = {h: float(v.mean()) for h, v in curves.items()}
        if ctx.writer:
            for h, err_tensor in curves.items():
                ctx.writer.add_scalar(f"rollout/mse_h{h:02d}", float(err_tensor.mean()), ctx.global_step)
        return True


class GradNormCallback(TrainCallback):
    """Tracks per-module gradient norms."""

    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = every_n_steps

    def on_step(self, ctx):
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        module_norms = {}
        for name, param in ctx.model.named_parameters():
            if param.grad is None:
                continue
            module_name = name.split(".")[0]
            if module_name not in module_norms:
                module_norms[module_name] = 0.0
            module_norms[module_name] += param.grad.data.norm(2).item() ** 2
        grad_norms = {k: v ** 0.5 for k, v in module_norms.items()}
        ctx.extras["grad_norms"] = grad_norms
        total_norm = sum(v ** 2 for v in grad_norms.values()) ** 0.5
        grad_norms["total"] = total_norm
        if ctx.writer:
            for module_name, norm in grad_norms.items():
                ctx.writer.add_scalar(f"grad/{module_name}", norm, ctx.global_step)
        return True
