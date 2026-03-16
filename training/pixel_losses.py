# training/pixel_losses.py
"""Loss functions for pixel world model training.

Separate from state-space losses (training/losses.py) because pixel
models have no delta normalization, no NormStats, and use different
loss structures (reconstruction + KL for VAE, latent MSE for dynamics).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 0.0001
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss: reconstruction + beta-weighted KL divergence.

    Returns:
        total_loss, recon_loss, kl_loss (all scalar tensors)
    """
    recon_loss = F.mse_loss(recon, target)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = kl_per_sample.mean()
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def latent_dynamics_loss(z_pred: torch.Tensor,
                         z_target: torch.Tensor) -> torch.Tensor:
    """MSE loss in latent space for dynamics prediction."""
    return F.mse_loss(z_pred, z_target)
