"""Shared test fixtures. Synthetic episodes for fast tests."""
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def episode_dir(tmp_path):
    """Create a directory with 10 small synthetic .npz episodes.

    Each episode: 50 steps, 15D states (8 kinematic + 7 constant physics),
    2D actions in [-1, 1], random rewards.
    """
    rng = np.random.default_rng(42)
    physics = np.array([-10.0, 13.0, 0.6, 5.0, 0.0, 15.0, 1.5], dtype=np.float32)

    for i in range(10):
        T = 50
        kinematic = rng.standard_normal((T + 1, 8)).astype(np.float32) * 0.1
        # Make kinematic states a random walk so deltas are small
        kinematic = np.cumsum(kinematic * 0.02, axis=0).astype(np.float32)
        physics_block = np.tile(physics, (T + 1, 1))
        states = np.concatenate([kinematic, physics_block], axis=1)
        actions = rng.uniform(-1, 1, (T, 2)).astype(np.float32)
        rewards = rng.standard_normal(T).astype(np.float32)
        metadata = json.dumps({"seed": i, "physics_config": {
            "gravity": -10.0, "main_engine_power": 13.0,
            "side_engine_power": 0.6, "lander_density": 5.0,
            "angular_damping": 0.0, "wind_power": 15.0,
            "turbulence_power": 1.5,
        }})
        np.savez(
            tmp_path / f"episode_{i:03d}.npz",
            states=states, actions=actions, rewards=rewards,
            metadata_json=metadata,
        )
    return tmp_path
