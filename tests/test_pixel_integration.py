# tests/test_pixel_integration.py
"""Integration tests for PixelWorldModel (VAE + dynamics combined)."""
import torch
import pytest
from models.pixel_vae import PixelVAE
from models.pixel_dynamics import LatentDynamicsModel
from models.pixel_world_model import PixelWorldModel


class TestPixelWorldModel:
    """End-to-end tests for combined pixel world model."""

    @pytest.fixture
    def model(self):
        vae = PixelVAE(in_channels=1, latent_dim=32, frame_size=84,
                       channels=[16, 32, 64, 128])
        dynamics = LatentDynamicsModel(latent_dim=32, action_dim=2, hidden_size=64)
        return PixelWorldModel(vae, dynamics)

    def test_predict_next_teacher_forced(self, model):
        """predict_next encodes real frame, predicts, decodes."""
        frame = torch.rand(2, 1, 84, 84)
        action = torch.randn(2, 2)
        pred_frame, z_next, hidden = model.predict_next(frame, action)
        assert pred_frame.shape == (2, 1, 84, 84)
        assert z_next.shape == (2, 32)

    def test_dream_autoregressive(self, model):
        """dream produces frame sequence from seed."""
        seed = torch.rand(2, 1, 84, 84)
        actions = torch.randn(2, 5, 2)
        frames = model.dream(seed, actions)
        assert frames.shape == (2, 6, 1, 84, 84)

    def test_dream_from_latent(self, model):
        """dream_from_latent operates purely in latent space."""
        z_seed = torch.randn(2, 32)
        actions = torch.randn(2, 5, 2)
        frames, z_seq = model.dream_from_latent(z_seed, actions)
        assert frames.shape == (2, 6, 1, 84, 84)
        assert z_seq.shape == (2, 6, 32)

    def test_encode_decode_roundtrip(self, model):
        """Encode then decode preserves shape."""
        frame = torch.rand(2, 1, 84, 84)
        z = model.encode(frame)
        recon = model.decode(z)
        assert recon.shape == frame.shape

    def test_4channel_stacked_dream(self):
        """Dream with 4-channel stacked frames uses frame buffer."""
        vae = PixelVAE(in_channels=4, latent_dim=32, frame_size=84,
                       channels=[16, 32, 64, 128])
        dynamics = LatentDynamicsModel(latent_dim=32, action_dim=2, hidden_size=64)
        model = PixelWorldModel(vae, dynamics)

        seed = torch.rand(1, 4, 84, 84)
        actions = torch.randn(1, 3, 2)
        frames = model.dream(seed, actions)
        assert frames.shape == (1, 4, 4, 84, 84)
