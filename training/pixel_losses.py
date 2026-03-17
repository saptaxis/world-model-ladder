# training/pixel_losses.py
"""Loss functions for pixel world model training.

Separate from state-space losses (training/losses.py) because pixel
models have no delta normalization, no NormStats, and use different
loss structures (reconstruction + KL for VAE, latent MSE for dynamics).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _foreground_weight_mask(target: torch.Tensor, fg_weight: float,
                            lo: float = 0.04, hi: float = 0.78) -> torch.Tensor:
    """Compute per-pixel weight mask from target frame.

    Pixels in the foreground intensity band (between black sky and white
    terrain) get fg_weight multiplier. Background pixels get weight 1.

    For grayscale [0,1]: black sky < lo (~10/255), white terrain > hi (~200/255).
    Everything in between is lander, legs, flames, flags.

    Args:
        target: (B, C, H, W) in [0, 1]
        fg_weight: multiplier for foreground pixels
        lo: lower intensity threshold (below = sky)
        hi: upper intensity threshold (above = terrain)

    Returns:
        (B, C, H, W) weight tensor, same shape as target
    """
    # Foreground: pixels with intensity in (lo, hi)
    fg_mask = (target > lo) & (target < hi)
    weights = torch.ones_like(target)
    weights[fg_mask] = fg_weight
    return weights


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.0001, fg_weight: float = 1.0,
             state_pred: torch.Tensor | None = None,
             state_target: torch.Tensor | None = None,
             state_weight: float = 0.0,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss: foreground-weighted reconstruction + KL + auxiliary state prediction.

    When fg_weight > 1, pixels in the foreground band (lander, flames,
    flags — between black sky and white terrain) are upweighted in the
    reconstruction loss.

    When state_pred and state_target are provided with state_weight > 0,
    adds MSE between predicted and ground-truth kinematic state. This
    forces the latent space to encode physical state (position, velocity,
    angle), giving z spatial meaning and preventing discontinuous jumps.

    Args:
        recon: reconstructed frames (B, C, H, W)
        target: original frames (B, C, H, W)
        mu: encoder mean (B, latent_dim)
        logvar: encoder log-variance (B, latent_dim)
        beta: KL weight
        fg_weight: foreground pixel weight (1.0 = uniform MSE)
        state_pred: predicted kinematic state (B, state_dim) or None
        state_target: ground truth kinematic state (B, state_dim) or None
        state_weight: weight for state prediction loss (0.0 = disabled)

    Returns:
        total_loss, recon_loss, kl_loss, state_loss (all scalar tensors)
    """
    if fg_weight > 1.0:
        weights = _foreground_weight_mask(target, fg_weight)
        recon_loss = (weights * (recon - target).pow(2)).mean()
    else:
        recon_loss = F.mse_loss(recon, target)

    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = kl_per_sample.mean()

    state_loss = torch.tensor(0.0, device=recon.device)
    if state_pred is not None and state_target is not None and state_weight > 0:
        state_loss = F.mse_loss(state_pred, state_target)

    total = recon_loss + beta * kl_loss + state_weight * state_loss
    return total, recon_loss, kl_loss, state_loss


def latent_dynamics_loss(z_pred: torch.Tensor,
                         z_target: torch.Tensor) -> torch.Tensor:
    """MSE loss in latent space for dynamics prediction."""
    return F.mse_loss(z_pred, z_target)
