import pytest
import torch
from models.base import WorldModel


def test_base_class_initial_state():
    """Stateless default returns None."""
    model = WorldModel()
    assert model.initial_state(batch_size=4) is None


def test_base_class_step_raises():
    model = WorldModel()
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    try:
        model.step(obs, action)
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass


from models.linear import LinearModel


def test_linear_output_shape():
    model = LinearModel(state_dim=8, action_dim=2)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms is None


def test_linear_is_world_model():
    model = LinearModel(state_dim=8, action_dim=2)
    assert isinstance(model, WorldModel)


def test_linear_gradient_flows():
    model = LinearModel(state_dim=8, action_dim=2)
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, _ = model.step(obs, action)
    loss = delta.pow(2).mean()
    loss.backward()
    assert model.linear.weight.grad is not None
    assert model.linear.weight.grad.abs().sum() > 0


from models.mlp import MLPModel


def test_mlp_output_shape():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[64, 64])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, ms = model.step(obs, action)
    assert delta.shape == (4, 8)
    assert ms is None


def test_mlp_is_world_model():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32])
    assert isinstance(model, WorldModel)


def test_mlp_different_architectures():
    """Various hidden_dims configs produce correct output."""
    for dims in [[64], [128, 128], [256, 256, 256]]:
        model = MLPModel(state_dim=8, action_dim=2, hidden_dims=dims)
        delta, _ = model.step(torch.randn(2, 8), torch.randn(2, 2))
        assert delta.shape == (2, 8)


def test_mlp_gradient_flows_through_all_layers():
    model = MLPModel(state_dim=8, action_dim=2, hidden_dims=[32, 32])
    obs = torch.randn(4, 8)
    action = torch.randn(4, 2)
    delta, _ = model.step(obs, action)
    loss = delta.pow(2).mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
