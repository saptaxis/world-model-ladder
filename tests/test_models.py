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
