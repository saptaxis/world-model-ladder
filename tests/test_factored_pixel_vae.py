"""Tests for FactoredPixelVAE with [z_kin, z_ctx] split."""
import torch
import pytest
from models.factored_pixel_vae import FactoredPixelVAE
from models.pixel_vae import PixelVAE


class TestFactoredPixelVAEConcat:
    @pytest.fixture
    def model(self):
        return FactoredPixelVAE(
            in_channels=1, latent_dim=16, frame_size=64,
            channels=[4, 8, 16, 32],
            kin_targets=[0, 1, 2, 3, 4, 5],  # all 6 kinematic dims
            decoder_type="concat",
        )

    def test_inherits_from_pixel_vae(self, model):
        """Must inherit from PixelVAE for PixelWorldModel compatibility."""
        assert isinstance(model, PixelVAE)

    def test_forward_returns_4_tuple(self, model):
        """forward() must return (recon, mu, logvar, state_pred) like PixelVAE."""
        x = torch.rand(2, 1, 64, 64)
        recon, mu, logvar, state_pred = model(x)
        assert recon.shape == (2, 1, 64, 64)
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
        # state_pred is z_kin slice (6 dims for all-kinematic targets)
        assert state_pred.shape == (2, 6)

    def test_predict_state_returns_z_kin_slice(self, model):
        """predict_state returns z[:, kin_indices] — a slice, not a head."""
        z = torch.randn(4, 16)
        state = model.predict_state(z)
        assert state.shape == (4, 6)
        # Should be a direct slice of z, not transformed
        assert torch.allclose(state, z[:, :6])

    def test_state_dim_property(self, model):
        """state_dim returns len(kin_targets) for compatibility checks."""
        assert model.state_dim == 6

    def test_decode_accepts_single_z(self, model):
        """decode(z) takes single concatenated tensor — same signature as PixelVAE."""
        z = torch.randn(2, 16)
        frame = model.decode(z)
        assert frame.shape == (2, 1, 64, 64)

    def test_encode_decode_round_trip(self, model):
        """Encode-decode produces same shape as input."""
        x = torch.rand(2, 1, 64, 64)
        z = model.encode(x)
        recon = model.decode(z)
        assert recon.shape == x.shape

    def test_kin_targets_subset(self):
        """kin_targets=[4,5] (angle-only) produces 2-dim z_kin."""
        model = FactoredPixelVAE(
            in_channels=1, latent_dim=16, frame_size=64,
            channels=[4, 8, 16, 32],
            kin_targets=[4, 5],  # angle + ang_vel only
            decoder_type="concat",
        )
        z = torch.randn(2, 16)
        state = model.predict_state(z)
        assert state.shape == (2, 2)
        assert model.state_dim == 2
        # Should be dims 4 and 5 of z
        assert torch.allclose(state, z[:, [4, 5]])

    def test_kin_targets_ordering(self):
        """kin_targets=[2,3,4] selects specific z dims in order."""
        model = FactoredPixelVAE(
            in_channels=1, latent_dim=16, frame_size=64,
            channels=[4, 8, 16, 32],
            kin_targets=[2, 3, 4],  # vx, vy, angle
            decoder_type="concat",
        )
        z = torch.randn(2, 16)
        state = model.predict_state(z)
        assert torch.allclose(state, z[:, [2, 3, 4]])


class TestFactoredPixelVAEFiLM:
    @pytest.fixture
    def model(self):
        return FactoredPixelVAE(
            in_channels=1, latent_dim=16, frame_size=64,
            channels=[4, 8, 16, 32],
            kin_targets=[0, 1, 2, 3, 4, 5],
            decoder_type="film",
        )

    def test_forward_returns_4_tuple(self, model):
        x = torch.rand(2, 1, 64, 64)
        recon, mu, logvar, state_pred = model(x)
        assert recon.shape == (2, 1, 64, 64)
        assert state_pred.shape == (2, 6)

    def test_decode_accepts_single_z(self, model):
        z = torch.randn(2, 16)
        frame = model.decode(z)
        assert frame.shape == (2, 1, 64, 64)

    def test_film_modulation_changes_output(self, model):
        """Different z_kin with same z_ctx should produce different frames."""
        z1 = torch.randn(1, 16)
        z2 = z1.clone()
        # Change only z_kin dims (first 6)
        z2[0, :6] = z1[0, :6] + 2.0
        frame1 = model.decode(z1)
        frame2 = model.decode(z2)
        # Frames should differ because z_kin changed
        assert not torch.allclose(frame1, frame2, atol=1e-4)

    def test_same_z_kin_different_z_ctx_differs(self, model):
        """Different z_ctx with same z_kin should also produce different frames
        (z_ctx provides the base features that get modulated)."""
        z1 = torch.randn(1, 16)
        z2 = z1.clone()
        # Change only z_ctx dims (dims 6-15)
        z2[0, 6:] = torch.randn(10)
        frame1 = model.decode(z1)
        frame2 = model.decode(z2)
        assert not torch.allclose(frame1, frame2, atol=1e-4)

    def test_film_with_angle_only_targets(self):
        """FiLM works with subset kin_targets."""
        model = FactoredPixelVAE(
            in_channels=1, latent_dim=16, frame_size=64,
            channels=[4, 8, 16, 32],
            kin_targets=[4, 5],  # angle-only
            decoder_type="film",
        )
        z = torch.randn(2, 16)
        frame = model.decode(z)
        assert frame.shape == (2, 1, 64, 64)
        assert model.predict_state(z).shape == (2, 2)

    def test_checkpoint_round_trip(self, model, tmp_path):
        """Save and reload should produce identical forward outputs."""
        model.eval()
        x = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            out1 = model(x)

        # Save
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "model_type": "factored",
                "decoder_type": "film",
                "in_channels": 1,
                "latent_dim": 16,
                "frame_size": 64,
                "channels": [4, 8, 16, 32],
                "kin_targets": [0, 1, 2, 3, 4, 5],
            },
        }, ckpt_path)

        # Reload
        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg = ckpt["config"]
        model2 = FactoredPixelVAE(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            frame_size=cfg["frame_size"],
            channels=cfg["channels"],
            kin_targets=cfg["kin_targets"],
            decoder_type=cfg["decoder_type"],
        )
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()

        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1[0], out2[0], atol=1e-6)  # recon
        assert torch.allclose(out1[3], out2[3], atol=1e-6)  # state_pred
