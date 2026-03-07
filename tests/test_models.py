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
