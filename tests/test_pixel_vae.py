# tests/test_pixel_vae.py
"""Tests for PixelVAE model."""
import torch
import pytest
from models.pixel_vae import PixelVAE


class TestPixelVAE:
    """Test PixelVAE encoder-decoder."""

    def test_encode_returns_mu_logvar(self):
        """Encoder produces mu and logvar of correct shape."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        x = torch.randn(2, 1, 84, 84)
        mu, logvar = vae.encode_params(x)
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_decode_reconstructs_shape(self):
        """Decoder output matches input spatial dims and channels."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        z = torch.randn(2, 64)
        recon = vae.decode(z)
        assert recon.shape == (2, 1, 84, 84)

    def test_forward_returns_recon_mu_logvar(self):
        """Forward pass returns reconstruction, mu, logvar."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        x = torch.randn(2, 1, 84, 84)
        recon, mu, logvar = vae(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_encode_deterministic_at_eval(self):
        """encode() returns mu (no sampling) for deterministic inference."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        vae.eval()
        x = torch.randn(1, 1, 84, 84)
        z = vae.encode(x)
        assert z.shape == (1, 64)
        # Should be deterministic
        z2 = vae.encode(x)
        assert torch.allclose(z, z2)

    def test_4channel_input(self):
        """VAE works with 4-channel (stacked frames) input."""
        vae = PixelVAE(in_channels=4, latent_dim=64, frame_size=84)
        x = torch.randn(2, 4, 84, 84)
        recon, mu, logvar = vae(x)
        assert recon.shape == (2, 4, 84, 84)

    def test_128_resolution(self):
        """VAE works at 128x128 resolution."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=128)
        x = torch.randn(2, 1, 128, 128)
        recon, mu, logvar = vae(x)
        assert recon.shape == (2, 1, 128, 128)

    def test_custom_channels(self):
        """VAE works with custom channel config."""
        vae = PixelVAE(in_channels=1, latent_dim=32,
                       frame_size=84, channels=[16, 32, 64, 128])
        x = torch.randn(2, 1, 84, 84)
        recon, mu, logvar = vae(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 32)

    def test_output_range_sigmoid(self):
        """Decoder output is in [0, 1] due to sigmoid."""
        vae = PixelVAE(in_channels=1, latent_dim=64, frame_size=84)
        vae.eval()
        x = torch.randn(2, 1, 84, 84).clamp(0, 1)
        recon, _, _ = vae(x)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0
