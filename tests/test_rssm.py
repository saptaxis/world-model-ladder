import torch
from models.rssm_state import RSSMState


def test_rssm_state_creation():
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
    )
    assert state.deter.shape == (4, 200)
    assert state.stoch.shape == (4, 30)
    assert state.prior_logits is None
    assert state.posterior_logits is None


def test_rssm_state_with_distribution_params():
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
        prior_logits=torch.randn(4, 30),
        posterior_logits=torch.randn(4, 30),
    )
    assert state.prior_logits is not None
    assert state.posterior_logits is not None


def test_rssm_state_detach():
    state = RSSMState(
        deter=torch.randn(4, 200, requires_grad=True),
        stoch=torch.randn(4, 30, requires_grad=True),
        prior_logits=torch.randn(4, 30, requires_grad=True),
        posterior_logits=torch.randn(4, 30, requires_grad=True),
    )
    detached = state.detach()
    assert not detached.deter.requires_grad
    assert not detached.stoch.requires_grad
    assert not detached.prior_logits.requires_grad
    assert not detached.posterior_logits.requires_grad


def test_rssm_state_to_device():
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
    )
    moved = state.to("cpu")
    assert moved.deter.device == torch.device("cpu")
    assert moved.stoch.device == torch.device("cpu")


def test_rssm_state_concat():
    """Feature vector is [deter, stoch]."""
    state = RSSMState(
        deter=torch.randn(4, 200),
        stoch=torch.randn(4, 30),
    )
    feat = state.features()
    assert feat.shape == (4, 230)
