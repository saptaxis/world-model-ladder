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
    PixelDynamicsValidationCallback,
    DreamGridCallback,
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


# ---------------------------------------------------------------------------
# PixelDynamicsValidationCallback tests
# ---------------------------------------------------------------------------

class TestPixelDynamicsValidationCallback:
    def test_latent_mse_mode(self):
        """Validation works with default latent_mse mode."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}},
        )
        # val_loader yields (frames_batch, actions_batch)
        # latent_mse uses predict_sequence which expects T actions for T+1 frames
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 10, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
        )
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_multi_step_mode(self):
        """Validation works with multi_step_latent mode."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}},
        )
        # multi_step expects T-1 actions for T frames
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 9, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
            training_mode="multi_step_latent",
            rollout_k=4,
        )
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_elbo_mode_with_rssm(self):
        """Validation works with latent_elbo mode and RSSM dynamics."""
        from models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}},
        )
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 9, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
            training_mode="latent_elbo",
            rollout_k=4, kl_weight=1.0,
        )
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        assert "val_loss" in ctx.extras

    def test_checkpoint_includes_model_type(self, tmp_path):
        """Saved checkpoint includes model_type from extras."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ckpt_dir = str(tmp_path / "ckpt")
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={"config": {}, "model_type": "rssm"},
        )
        val_data = [(torch.rand(2, 10, 1, 64, 64), torch.randn(2, 10, 2))]
        cb = PixelDynamicsValidationCallback(
            val_loader=val_data, vae=vae,
            every_n_steps=1, patience=5,
            checkpoint_dir=ckpt_dir,
        )
        cb.on_train_start(ctx)
        ctx.global_step = 1
        cb.on_step(ctx)
        # Load checkpoint and verify model_type is saved
        ckpt = torch.load(Path(ckpt_dir) / "best.pt", weights_only=False)
        assert ckpt["model_type"] == "rssm"


# ---------------------------------------------------------------------------
# DreamGridCallback tests
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal dataset returning (frames, actions) tuples for testing."""
    def __init__(self, T=10, C=1, H=64, W=64, action_dim=2):
        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.action_dim = action_dim

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        # (T, C, H, W), (T-1, action_dim) — matches episode dataset format
        return torch.rand(self.T, self.C, self.H, self.W), \
               torch.randn(self.T - 1, self.action_dim)


class TestDreamGridCallback:
    def test_dream_grid_fires(self):
        """DreamGridCallback fires without error (no writer = no-op image)."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={},
        )
        cb = DreamGridCallback(vae=vae, val_dataset=_FakeDataset(),
                               n_episodes=2, every_n_steps=1)
        ctx.global_step = 1
        result = cb.on_step(ctx)
        # No writer, so grid is skipped — but should still return True
        assert result is True

    def test_dream_grid_with_rssm(self):
        """DreamGridCallback works with RSSM dynamics model."""
        from models.pixel_rssm import LatentRSSM
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentRSSM(latent_dim=16, action_dim=2, deter_dim=32,
                              stoch_dim=8, hidden_dim=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=None,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={},
        )
        cb = DreamGridCallback(vae=vae, val_dataset=_FakeDataset(),
                               n_episodes=2, every_n_steps=1)
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True

    def test_dream_grid_produces_image(self):
        """DreamGridCallback produces image grid when writer is provided."""
        vae = PixelVAE(in_channels=1, latent_dim=16, frame_size=64,
                       channels=[8, 16, 32, 64])
        vae.eval()
        dynamics = LatentDynamicsModel(latent_dim=16, action_dim=2,
                                       hidden_size=32)
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)

        # Mock writer to capture add_image calls
        class MockWriter:
            def __init__(self):
                self.images = {}
            def add_image(self, tag, img, step):
                self.images[tag] = (img, step)

        writer = MockWriter()
        ctx = CallbackContext(
            model=dynamics, optimizer=optimizer, writer=writer,
            global_step=0, epoch=0, run_dir="/tmp", device="cpu",
            extras={},
        )
        cb = DreamGridCallback(vae=vae, val_dataset=_FakeDataset(),
                               n_episodes=2, every_n_steps=1)
        ctx.global_step = 1
        result = cb.on_step(ctx)
        assert result is True
        # Verify image was logged
        assert "dynamics/dream_grid" in writer.images
        img, step = writer.images["dynamics/dream_grid"]
        assert step == 1
        # Grid should be a 3D tensor (C, H, W)
        assert img.ndim == 3
