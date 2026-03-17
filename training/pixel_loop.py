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
                          fg_weight: float = 1.0, state_weight: float = 0.0,
                          device: str = "cpu", max_grad_norm: float = 1.0,
                          ctx=None, callbacks=None) -> dict:
    """One VAE training epoch with callback dispatch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_state = 0.0
    n_batches = 0
    stop_requested = False

    for batch in train_loader:
        if ctx is not None:
            ctx.global_step += 1

        # batch is either a tensor (frames only) or tuple (frames, states)
        if isinstance(batch, torch.Tensor):
            x = batch.to(device)
            state_target = None
        else:
            x = batch[0].to(device)
            state_target = batch[1].to(device) if len(batch) > 1 else None

        recon, mu, logvar, state_pred = model(x)
        loss, recon_loss, kl_loss, state_loss = vae_loss(
            recon, x, mu, logvar, beta, fg_weight=fg_weight,
            state_pred=state_pred, state_target=state_target,
            state_weight=state_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if ctx is not None and callbacks:
            ctx.extras["train_loss_step"] = loss.item()
            ctx.extras["recon_loss_step"] = recon_loss.item()
            ctx.extras["kl_loss_step"] = kl_loss.item()
            ctx.extras["state_loss_step"] = state_loss.item()
            if ctx.writer:
                ctx.writer.add_scalar("train/loss", loss.item(), ctx.global_step)
                ctx.writer.add_scalar("train/recon_loss", recon_loss.item(), ctx.global_step)
                ctx.writer.add_scalar("train/kl_loss", kl_loss.item(), ctx.global_step)
                if state_weight > 0:
                    ctx.writer.add_scalar("train/state_loss", state_loss.item(), ctx.global_step)
            for cb in callbacks:
                if cb.on_step(ctx) is False:
                    stop_requested = True
                    break

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_state += state_loss.item()
        n_batches += 1

        if stop_requested:
            break

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "recon_loss": total_recon / max(n_batches, 1),
        "kl_loss": total_kl / max(n_batches, 1),
        "state_loss": total_state / max(n_batches, 1),
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
