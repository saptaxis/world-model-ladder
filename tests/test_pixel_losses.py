# tests/test_pixel_losses.py
"""Tests for pixel-space loss functions."""
import torch
import pytest
from training.pixel_losses import vae_loss, latent_dynamics_loss


class TestVAELoss:
    def test_zero_reconstruction_error(self):
        """Perfect reconstruction -> loss is just KL."""
        x = torch.rand(4, 1, 84, 84)
        recon = x.clone()
        mu = torch.zeros(4, 64)
        logvar = torch.zeros(4, 64)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=1.0)
        assert recon_loss.item() < 1e-6
        assert kl_loss.item() < 1e-6

    def test_beta_scales_kl(self):
        x = torch.rand(4, 1, 84, 84)
        recon = x + 0.1 * torch.randn_like(x)
        mu = torch.randn(4, 64)
        logvar = torch.randn(4, 64)
        loss_b1, _, kl1 = vae_loss(recon, x, mu, logvar, beta=1.0)
        loss_b01, _, kl01 = vae_loss(recon, x, mu, logvar, beta=0.1)
        assert torch.allclose(kl1, kl01)
        assert loss_b1 > loss_b01


class TestLatentDynamicsLoss:
    def test_mse_in_latent_space(self):
        z_pred = torch.randn(4, 10, 64)
        z_target = z_pred.clone()
        loss = latent_dynamics_loss(z_pred, z_target)
        assert loss.item() < 1e-6

    def test_nonzero_for_different(self):
        z_pred = torch.randn(4, 10, 64)
        z_target = torch.randn(4, 10, 64)
        loss = latent_dynamics_loss(z_pred, z_target)
        assert loss.item() > 0
