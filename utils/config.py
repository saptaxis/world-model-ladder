"""RunConfig dataclass, YAML serialization, and run name generation."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    # Architecture
    arch: str                           # "linear" | "mlp" | "gru" | "rssm"
    arch_params: dict = field(default_factory=dict)

    # Prediction
    prediction: str = "delta"           # "absolute" | "delta"

    # Training
    training_mode: str = "single_step"  # "single_step" | "multi_step" | "scheduled_sampling"
    rollout_k: int = 1
    sampling_start: float = 0.0
    sampling_end: float = 0.5
    curriculum: bool = False
    kl_weight: float = 1.0              # KL weight for ELBO loss

    # Data
    data_mix: str = "policy"            # "policy" | "policy_primitives"
    data_path: str = ""                 # required — validated in __post_init__

    # Environment
    state_dim: int = 8
    action_dim: int = 2

    # Training hyperparams
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    seq_len: int = 50
    val_fraction: float = 0.1

    # Output
    run_dir: str = "runs/"
    suffix: str = ""

    def __post_init__(self):
        if not self.data_path:
            raise ValueError("data_path is required (got empty string)")

    def save(self, path: str | Path):
        """Save config as YAML."""
        d = dataclasses.asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load config from YAML."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)


def generate_run_name(config: RunConfig) -> str:
    """Auto-generate run name from config axes.

    Format: {arch}-{prediction}-{training_mode}_k{rollout_k}-{data_mix}[--{suffix}]
    """
    name = f"{config.arch}-{config.prediction}-{config.training_mode}_k{config.rollout_k}-{config.data_mix}"
    if config.suffix:
        name += f"--{config.suffix}"
    return name


def load_config(path: str, overrides: dict | None = None) -> RunConfig:
    """Load config from YAML with optional field overrides."""
    cfg = RunConfig.load(path)
    if overrides:
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise ValueError(f"Unknown config field: {key}")
    return cfg
