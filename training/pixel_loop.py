# training/pixel_loop.py
"""Training loops for pixel world models.

Follows the same callback dispatch pattern as training/loop.py so that
existing model-agnostic callbacks (CheckpointCallback, GradNormCallback,
NaNDetectionCallback, ProgressCallback, PlotExportCallback) work
without modification.
"""
from __future__ import annotations

import time

import torch

from training.pixel_losses import vae_loss, latent_dynamics_loss


def pixel_vae_train_epoch(model, train_loader, optimizer, beta: float = 0.0001,
                          device: str = "cpu", max_grad_norm: float = 1.0,
                          ctx=None, callbacks=None) -> dict:
    """One VAE training epoch with callback dispatch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0
    stop_requested = False

    for batch in train_loader:
        if ctx is not None:
            ctx.global_step += 1

        x = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
        recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if ctx is not None and callbacks:
            ctx.extras["train_loss_step"] = loss.item()
            ctx.extras["recon_loss_step"] = recon_loss.item()
            ctx.extras["kl_loss_step"] = kl_loss.item()
            if ctx.writer:
                ctx.writer.add_scalar("train/loss", loss.item(), ctx.global_step)
                ctx.writer.add_scalar("train/recon_loss", recon_loss.item(), ctx.global_step)
                ctx.writer.add_scalar("train/kl_loss", kl_loss.item(), ctx.global_step)
            for cb in callbacks:
                if cb.on_step(ctx) is False:
                    stop_requested = True
                    break

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

        if stop_requested:
            break

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "recon_loss": total_recon / max(n_batches, 1),
        "kl_loss": total_kl / max(n_batches, 1),
        "stop_requested": stop_requested,
    }


def pixel_dynamics_train_epoch(model_dynamics, vae, train_loader, optimizer,
                               sampling_prob: float = 0.0,
                               device: str = "cpu", max_grad_norm: float = 1.0,
                               ctx=None, callbacks=None) -> dict:
    """One dynamics training epoch with frozen VAE and callback dispatch."""
    model_dynamics.train()
    total_loss = 0.0
    n_batches = 0
    stop_requested = False

    for frames_batch, actions_batch in train_loader:
        if ctx is not None:
            ctx.global_step += 1

        B, T, C, H, W = frames_batch.shape
        frames_flat = frames_batch.reshape(B * T, C, H, W).to(device)
        actions = actions_batch.to(device)

        with torch.no_grad():
            z_all = vae.encode(frames_flat)
        latent_dim = z_all.shape[-1]
        z_seq = z_all.reshape(B, T, latent_dim)

        z_pred, _ = model_dynamics.predict_sequence(
            z_seq, actions, teacher_forcing=1.0 - sampling_prob)

        loss = latent_dynamics_loss(z_pred[:, :-1], z_seq[:, 1:])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_dynamics.parameters(), max_grad_norm)

        if ctx is not None and callbacks:
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
