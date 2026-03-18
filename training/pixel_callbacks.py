# training/pixel_callbacks.py
"""Pixel-specific training callbacks.

Reuses the CallbackContext and TrainCallback base from training/callbacks.py.
Model-agnostic callbacks (CheckpointCallback, GradNormCallback,
NaNDetectionCallback, ProgressCallback, PlotExportCallback) work directly
with pixel training loops -- no changes needed.

These callbacks handle pixel-specific concerns: VAE validation with
reconstruction+KL loss, reconstruction grid logging, dream sequence logging.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training.callbacks import TrainCallback, CallbackContext
from training.pixel_losses import vae_loss, latent_dynamics_loss


class PixelVAEValidationCallback(TrainCallback):
    """VAE validation with early stopping and best-checkpoint saving."""

    def __init__(self, val_loader, beta: float = 0.0001,
                 fg_weight: float = 1.0, state_weight: float = 0.0,
                 every_n_steps: int = 500, patience: int = 10,
                 checkpoint_dir: str | None = None):
        self.val_loader = val_loader
        self.beta = beta
        self.fg_weight = fg_weight
        self.state_weight = state_weight
        self.every_n_steps = every_n_steps
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.last_val_step = -1

    def on_train_start(self, ctx: CallbackContext):
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_step(self, ctx: CallbackContext) -> bool:
        if ctx.global_step == 0:
            return True
        if ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.global_step == self.last_val_step:
            return True
        self.last_val_step = ctx.global_step

        ctx.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_state = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, torch.Tensor):
                    x = batch.to(ctx.device)
                    state_target = None
                else:
                    x = batch[0].to(ctx.device)
                    state_target = batch[1].to(ctx.device) if len(batch) > 1 else None
                recon, mu, logvar, state_pred = ctx.model(x)
                loss, recon_l, kl_l, state_l = vae_loss(
                    recon, x, mu, logvar, self.beta,
                    fg_weight=self.fg_weight,
                    state_pred=state_pred, state_target=state_target,
                    state_weight=self.state_weight)
                total_loss += loss.item()
                total_recon += recon_l.item()
                total_kl += kl_l.item()
                total_state += state_l.item()
                n += 1
        ctx.model.train()

        val_loss = total_loss / max(n, 1)
        ctx.extras["val_loss"] = val_loss

        if ctx.writer:
            ctx.writer.add_scalar("val/loss", val_loss, ctx.global_step)
            ctx.writer.add_scalar("val/recon_loss", total_recon / max(n, 1), ctx.global_step)
            ctx.writer.add_scalar("val/kl_loss", total_kl / max(n, 1), ctx.global_step)
            if self.state_weight > 0:
                ctx.writer.add_scalar("val/state_loss", total_state / max(n, 1), ctx.global_step)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.checkpoint_dir:
                torch.save({
                    "model_state_dict": ctx.model.state_dict(),
                    "optimizer_state_dict": ctx.optimizer.state_dict(),
                    "epoch": ctx.epoch,
                    "global_step": ctx.global_step,
                    "val_loss": val_loss,
                    "config": ctx.extras.get("config", {}),
                }, Path(self.checkpoint_dir) / "best.pt")
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"Early stopping at step {ctx.global_step} "
                  f"(patience {self.patience}, best={self.best_val_loss:.6f})")
            return False

        return True


class ReconGridCallback(TrainCallback):
    """Log reconstruction grid (original vs reconstructed) to TensorBoard."""

    def __init__(self, val_loader, every_n_steps: int = 500):
        self.val_loader = val_loader
        self.every_n_steps = every_n_steps

    def on_step(self, ctx: CallbackContext) -> bool:
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.writer is None:
            return True

        # Sample random batch from val each time
        batch = next(iter(self.val_loader))
        x = batch.to(ctx.device) if isinstance(batch, torch.Tensor) else batch[0].to(ctx.device)
        x = x[:8]

        ctx.model.eval()
        with torch.no_grad():
            recon, _, _, _ = ctx.model(x)
            pairs = torch.stack([x, recon], dim=1).reshape(-1, *x.shape[1:])
            try:
                from torchvision.utils import make_grid
                grid = make_grid(pairs, nrow=2, normalize=False)
                ctx.writer.add_image("vae/recon_grid", grid, ctx.global_step)
            except ImportError:
                pass
        ctx.model.train()
        return True


class PixelDynamicsValidationCallback(TrainCallback):
    """Dynamics validation with early stopping."""

    def __init__(self, val_loader, vae: nn.Module,
                 every_n_steps: int = 500, patience: int = 10,
                 checkpoint_dir: str | None = None):
        self.val_loader = val_loader
        self.vae = vae
        self.every_n_steps = every_n_steps
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.last_val_step = -1

    def on_train_start(self, ctx):
        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def on_step(self, ctx) -> bool:
        if ctx.global_step == 0:
            return True
        if ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.global_step == self.last_val_step:
            return True
        self.last_val_step = ctx.global_step

        ctx.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for frames_batch, actions_batch in self.val_loader:
                B, T, C, H, W = frames_batch.shape
                frames_flat = frames_batch.reshape(B * T, C, H, W).to(ctx.device)
                actions = actions_batch.to(ctx.device)
                z_all = self.vae.encode(frames_flat)
                latent_dim = z_all.shape[-1]
                z_seq = z_all.reshape(B, T, latent_dim)
                z_pred, _ = ctx.model.predict_sequence(
                    z_seq, actions, teacher_forcing=0.0)
                loss = latent_dynamics_loss(z_pred[:, :-1], z_seq[:, 1:])
                total_loss += loss.item()
                n += 1
        ctx.model.train()

        val_loss = total_loss / max(n, 1)
        ctx.extras["val_loss"] = val_loss

        if ctx.writer:
            ctx.writer.add_scalar("val/loss", val_loss, ctx.global_step)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.checkpoint_dir:
                torch.save({
                    "model_state_dict": ctx.model.state_dict(),
                    "optimizer_state_dict": ctx.optimizer.state_dict(),
                    "epoch": ctx.epoch,
                    "global_step": ctx.global_step,
                    "val_loss": val_loss,
                    "config": ctx.extras.get("config", {}),
                    "vae_checkpoint": ctx.extras.get("vae_checkpoint", ""),
                }, Path(self.checkpoint_dir) / "best.pt")
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"Early stopping at step {ctx.global_step} "
                  f"(patience {self.patience}, best={self.best_val_loss:.6f})")
            return False

        return True


class DreamGridCallback(TrainCallback):
    """Log dream sequence to TensorBoard during dynamics training."""

    def __init__(self, vae: nn.Module, sample_frames: torch.Tensor,
                 sample_actions: torch.Tensor, every_n_steps: int = 1000):
        self.vae = vae
        self.sample_frames = sample_frames
        self.sample_actions = sample_actions
        self.every_n_steps = every_n_steps

    def on_step(self, ctx) -> bool:
        if ctx.global_step == 0 or ctx.global_step % self.every_n_steps != 0:
            return True
        if ctx.writer is None:
            return True

        ctx.model.eval()
        with torch.no_grad():
            seed = self.sample_frames.to(ctx.device)
            actions = self.sample_actions.to(ctx.device)
            z = self.vae.encode(seed)
            T = actions.size(1)

            z_seq = [z]
            hidden = None
            for t in range(T):
                z_next, hidden = ctx.model(z, actions[:, t], hidden)
                z_seq.append(z_next)
                z = z_next

            indices = list(range(0, len(z_seq), max(1, len(z_seq) // 6)))[:6]
            frames = []
            for i in indices:
                frame = self.vae.decode(z_seq[i])
                frames.append(frame)

            try:
                from torchvision.utils import make_grid
                grid = make_grid(torch.cat(frames, dim=0), nrow=len(frames),
                                 normalize=False)
                ctx.writer.add_image("dynamics/dream_grid", grid, ctx.global_step)
            except ImportError:
                pass
        ctx.model.train()
        return True
