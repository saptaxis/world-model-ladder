# tests/test_pixel_callbacks.py
"""Tests for pixel-specific callbacks."""
import tempfile
from pathlib import Path

import torch
import pytest

from models.pixel_vae import PixelVAE
from models.pixel_dynamics import LatentDynamicsModel
from training.callbacks import CallbackContext
from training.pixel_callbacks import (
    PixelVAEValidationCallback,
    ReconGridCallback,
)


@pytest.fixture
def vae():
    return PixelVAE(in_channels=1, latent_dim=16, frame_size=84,
                    channels=[8, 16, 32, 64])


@pytest.fixture
def ctx(vae, tmp_path):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    return CallbackContext(
        model=vae, optimizer=optimizer, writer=None,
        global_step=0, epoch=0, run_dir=str(tmp_path),
        device="cpu", extras={"config": {"test": True}},
    )


class TestPixelVAEValidationCallback:
    def test_triggers_at_interval(self, vae, ctx, tmp_path):
        """Callback runs validation at specified step interval."""
        val_data = [torch.rand(4, 1, 84, 84) for _ in range(2)]
        val_loader = val_data

        cb = PixelVAEValidationCallback(
            val_loader=val_loader, beta=0.0001,
            every_n_steps=10, patience=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cb.on_train_start(ctx)

        ctx.global_step = 1
        ctx.extras["train_loss_step"] = 0.5
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" not in ctx.extras

        ctx.global_step = 10
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_early_stopping(self, vae, ctx, tmp_path):
        """Callback returns False after patience exhausted."""
        val_data = [torch.rand(4, 1, 84, 84) for _ in range(2)]

        cb = PixelVAEValidationCallback(
            val_loader=val_data, beta=0.0001,
            every_n_steps=1, patience=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cb.on_train_start(ctx)

        cb.best_val_loss = 0.0
        for step in range(1, 10):
            ctx.global_step = step
            ctx.extras["train_loss_step"] = 1.0
            result = cb.on_step(ctx)
            if result is False:
                assert step <= 4
                return
        pytest.fail("Early stopping did not trigger")


class TestReconGridCallback:
    def test_logs_grid(self, vae, ctx):
        """ReconGridCallback produces reconstruction grid without error."""
        val_loader = [torch.rand(8, 1, 84, 84)]
        cb = ReconGridCallback(val_loader=val_loader, every_n_steps=5)

        ctx.global_step = 5
        result = cb.on_step(ctx)
        assert result is True
