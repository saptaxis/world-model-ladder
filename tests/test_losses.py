import torch
from models.mlp import MLPModel
from data.normalization import NormStats
from training.losses import single_step_loss, multi_step_loss


def _make_norm_stats():
    return NormStats(
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        delta_mean=torch.zeros(8),
        delta_std=torch.ones(8),
    )


def test_single_step_loss_finite():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    batch = (torch.randn(16, 8), torch.randn(16, 2), torch.randn(16, 8))
    loss = single_step_loss(model, batch, _make_norm_stats())
    assert loss.isfinite()
    assert loss.item() > 0


def test_single_step_loss_backward():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    batch = (torch.randn(16, 8), torch.randn(16, 2), torch.randn(16, 8))
    loss = single_step_loss(model, batch, _make_norm_stats())
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_multi_step_loss_finite():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    # batch for multi-step: (state_seq, action_seq) where state_seq is T+1 states
    state_seq = torch.randn(8, 11, 8)   # batch=8, T=10+1 states
    action_seq = torch.randn(8, 10, 2)  # T=10 actions
    batch = (state_seq, action_seq)
    loss = multi_step_loss(model, batch, _make_norm_stats(), k=5)
    assert loss.isfinite()
    assert loss.item() > 0


def test_multi_step_loss_backward():
    """Gradients flow through the k-step unrolled chain."""
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    state_seq = torch.randn(4, 11, 8)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = multi_step_loss(model, batch, _make_norm_stats(), k=5)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_multi_step_k1_close_to_single_step():
    """Multi-step with k=1 should give similar loss to single-step."""
    torch.manual_seed(42)
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    model.eval()

    states = torch.randn(16, 2, 8)  # just 2 states (for k=1)
    actions = torch.randn(16, 1, 2)

    ss_batch = (states[:, 0], actions[:, 0], states[:, 1] - states[:, 0])
    ms_batch = (states, actions)
    ns = _make_norm_stats()

    with torch.no_grad():
        ss = single_step_loss(model, ss_batch, ns)
        ms = multi_step_loss(model, ms_batch, ns, k=1)
    assert abs(ss.item() - ms.item()) < 0.01
