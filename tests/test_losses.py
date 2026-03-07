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


from models.gru import GRUModel


def test_multi_step_loss_threads_model_state():
    """Multi-step loss should thread model_state so GRU gets recurrence.

    Compare multi-step loss with GRU vs a version that manually resets
    hidden state each step. They should differ (proving state is threaded).
    """
    torch.manual_seed(42)
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    model.eval()
    state_seq = torch.randn(4, 6, 8)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = _make_norm_stats()

    with torch.no_grad():
        loss = multi_step_loss(model, batch, ns, k=5)

    # Manually compute loss WITHOUT threading state (old buggy behavior)
    from data.normalization import normalize, denormalize
    import torch.nn.functional as F
    s = state_seq[:, 0]
    pred_deltas_raw = []
    with torch.no_grad():
        for t in range(5):
            s_n = normalize(s, ns.state_mean, ns.state_std)
            delta_n, _ = model.step(s_n, action_seq[:, t])  # no model_state!
            delta_raw = denormalize(delta_n, ns.delta_mean, ns.delta_std)
            s = s + delta_raw
            pred_deltas_raw.append(delta_raw)
    pred_deltas_raw = torch.stack(pred_deltas_raw, dim=1)
    true_deltas = state_seq[:, 1:6] - state_seq[:, :5]
    true_deltas_n = normalize(true_deltas, ns.delta_mean, ns.delta_std)
    pred_deltas_n = normalize(pred_deltas_raw, ns.delta_mean, ns.delta_std)
    loss_no_state = F.mse_loss(pred_deltas_n, true_deltas_n)

    # These should differ because GRU hidden state evolves
    assert not torch.allclose(loss, loss_no_state), \
        "multi_step_loss should thread model_state for recurrent models"


def test_single_step_loss_gru():
    """Single-step loss works with GRU (ignores hidden state)."""
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    batch = (torch.randn(16, 8), torch.randn(16, 2), torch.randn(16, 8))
    loss = single_step_loss(model, batch, _make_norm_stats())
    assert loss.isfinite()


def test_multi_step_loss_gru():
    """Multi-step loss works with GRU, gradients flow through hidden state chain."""
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    state_seq = torch.randn(4, 11, 8)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = multi_step_loss(model, batch, _make_norm_stats(), k=5)
    assert loss.isfinite()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


from training.losses import scheduled_sampling_loss


def test_scheduled_sampling_loss_finite():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    state_seq = torch.randn(4, 11, 8)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = scheduled_sampling_loss(model, batch, _make_norm_stats(),
                                   k=5, sampling_prob=0.3)
    assert loss.isfinite()
    assert loss.item() > 0


def test_scheduled_sampling_loss_backward():
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    state_seq = torch.randn(4, 11, 8)
    action_seq = torch.randn(4, 10, 2)
    batch = (state_seq, action_seq)
    loss = scheduled_sampling_loss(model, batch, _make_norm_stats(),
                                   k=5, sampling_prob=0.3)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_scheduled_sampling_prob0_vs_multi_step():
    """With sampling_prob=0 (teacher-forced input) vs multi-step (open-loop input).

    These differ because multi_step feeds predicted states back while
    scheduled_sampling with prob=0 feeds true states. Both should produce
    finite losses. With GRU, both thread model_state.
    """
    torch.manual_seed(42)
    model = GRUModel(state_dim=8, action_dim=2, hidden_dim=16)
    model.eval()
    state_seq = torch.randn(4, 6, 8)
    action_seq = torch.randn(4, 5, 2)
    batch = (state_seq, action_seq)
    ns = _make_norm_stats()
    with torch.no_grad():
        ms_loss = multi_step_loss(model, batch, ns, k=5)
        ss_loss = scheduled_sampling_loss(model, batch, ns, k=5, sampling_prob=0.0)
    assert ms_loss.isfinite()
    assert ss_loss.isfinite()
    # They should differ because the state inputs differ (open-loop vs teacher-forced)
    # but both should be reasonable
    assert ms_loss.item() < 1000
    assert ss_loss.item() < 1000
