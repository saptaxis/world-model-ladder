"""Model construction from RunConfig."""
from __future__ import annotations

from utils.config import RunConfig
from models.linear import LinearModel
from models.mlp import MLPModel
from models.gru import GRUModel


def build_model(config: RunConfig):
    """Construct a WorldModel from config."""
    if config.arch == "linear":
        return LinearModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )
    elif config.arch == "mlp":
        params = config.arch_params
        return MLPModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=params.get("hidden_dims", [256, 256]),
            activation=params.get("activation", "relu"),
            dropout=params.get("dropout", 0.0),
        )
    elif config.arch == "gru":
        params = config.arch_params
        return GRUModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=params.get("hidden_dim", 128),
            num_layers=params.get("num_layers", 1),
            encoder_dims=params.get("encoder_dims"),
            decoder_dims=params.get("decoder_dims"),
        )
    else:
        raise ValueError(
            f"Unknown architecture: {config.arch}. "
            f"Available: linear, mlp, gru"
        )
