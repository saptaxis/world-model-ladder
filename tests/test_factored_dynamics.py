"""Tests for FactoredDynamicsModel with separate f_kin + f_ctx heads."""
import torch
import pytest
from models.factored_dynamics import FactoredDynamicsModel


class TestFactoredDynamics:
    @pytest.fixture
    def model(self):
        return FactoredDynamicsModel(
            latent_dim=16, action_dim=2, hidden_size=32,
            kin_dims=6,
        )

    def test_forward_returns_full_z(self, model):
        """forward() returns concatenated z_next (kin + ctx), same as GRU."""
        z = torch.randn(4, 16)
        action = torch.randn(4, 2)
        z_next, hidden = model(z, action)
        assert z_next.shape == (4, 16)  # full latent_dim, not split

    def test_rollout_shape(self, model):
        """rollout returns (B, T+1, latent_dim) — same contract as GRU."""
        z_start = torch.randn(4, 16)
        actions = torch.randn(4, 10, 2)
        z_seq, hidden = model.rollout(z_start, actions)
        assert z_seq.shape == (4, 11, 16)
        # Seed preserved at index 0
        assert torch.allclose(z_seq[:, 0], z_start)

    def test_initial_state(self, model):
        """initial_state returns proper hidden state."""
        state = model.initial_state(4, torch.device("cpu"))
        assert state is not None

    def test_kin_dims_separation(self, model):
        """f_kin and f_ctx operate on separate dim ranges."""
        z = torch.randn(1, 16)
        action = torch.randn(1, 2)
        z_next, _ = model(z, action)
        # The model should produce different z_next for different actions
        # (proving both heads are active, not just copying)
        action2 = torch.randn(1, 2)
        z_next2, _ = model(z, action2)
        # At minimum, z_kin dims should differ (action affects physics)
        assert not torch.allclose(z_next[:, :6], z_next2[:, :6], atol=1e-4)

    def test_f_ctx_receives_z_kin(self, model):
        """f_ctx sees z_kin — appearance depends on kinematics (position
        determines where lander pixels are)."""
        z = torch.randn(1, 16)
        action = torch.randn(1, 2)
        z_next_a, _ = model(z, action)
        # Modify only z_kin dims, keep z_ctx the same
        z2 = z.clone()
        z2[:, :6] = z[:, :6] + 2.0
        z_next_b, _ = model(z2, action)
        # z_ctx output should differ because f_ctx sees z_kin
        assert not torch.allclose(z_next_a[:, 6:], z_next_b[:, 6:], atol=1e-4)

    def test_predict_sequence_works(self, model):
        """predict_sequence for compatibility with latent_mse training mode."""
        z_seq = torch.randn(2, 10, 16)
        actions = torch.randn(2, 10, 2)
        z_pred, hidden = model.predict_sequence(z_seq, actions, teacher_forcing=1.0)
        assert z_pred.shape == (2, 10, 16)
