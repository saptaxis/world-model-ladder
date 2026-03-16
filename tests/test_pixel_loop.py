# tests/test_pixel_loop.py
"""Smoke tests for pixel training loops."""
import torch
import pytest
from models.pixel_vae import PixelVAE
from models.pixel_dynamics import LatentDynamicsModel
from training.callbacks import CallbackContext
from training.pixel_loop import pixel_vae_train_epoch, pixel_dynamics_train_epoch


class TestPixelVAETrainEpoch:
    def test_returns_loss_dict(self):
        """pixel_vae_train_epoch returns expected keys."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                       channels=[8, 16, 32, 64])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        loader = [torch.rand(4, 1, 84, 84) for _ in range(3)]
        result = pixel_vae_train_epoch(vae, loader, optimizer, beta=0.0001)
        assert "train_loss" in result
        assert "recon_loss" in result
        assert "kl_loss" in result
        assert result["train_loss"] > 0

    def test_callbacks_dispatched(self):
        """Callbacks receive on_step calls."""
        from training.callbacks import TrainCallback
        calls = []
        class CountCallback(TrainCallback):
            def on_step(self, ctx):
                calls.append(ctx.global_step)
                return True

        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                       channels=[8, 16, 32, 64])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        loader = [torch.rand(4, 1, 84, 84) for _ in range(3)]
        ctx = CallbackContext(model=vae, optimizer=optimizer, writer=None,
                              global_step=0, epoch=0, run_dir="/tmp",
                              device="cpu", extras={})
        pixel_vae_train_epoch(vae, loader, optimizer, ctx=ctx,
                              callbacks=[CountCallback()])
        assert len(calls) == 3


class TestPixelDynamicsTrainEpoch:
    def test_returns_loss_dict(self):
        """pixel_dynamics_train_epoch returns expected keys."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                       channels=[8, 16, 32, 64])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2, hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        loader = [(torch.rand(2, 10, 1, 84, 84), torch.randn(2, 10, 2))
                   for _ in range(2)]
        result = pixel_dynamics_train_epoch(dynamics, vae, loader, optimizer)
        assert "train_loss" in result
        assert result["train_loss"] > 0
