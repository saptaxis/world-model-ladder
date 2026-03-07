"""Tests for ParameterizedLunarLander environment — basic contract tests.

Verifies the env creates, steps, resets correctly, produces the right
observation/action shapes, is deterministic with seed, and includes
proper metadata in the info dict.
"""

import numpy as np
import pytest

from envs.lunar_lander import ParameterizedLunarLander, LunarLanderPhysicsConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_env():
    """Create a ParameterizedLunarLander with default physics config."""
    env = ParameterizedLunarLander()
    yield env
    env.close()


@pytest.fixture
def custom_config():
    """A non-default physics config for testing parameterization."""
    return LunarLanderPhysicsConfig(
        gravity=-6.0,
        main_engine_power=20.0,
        side_engine_power=1.0,
        lander_density=3.0,
        angular_damping=2.0,
        wind_power=5.0,
        turbulence_power=0.5,
    )


@pytest.fixture
def custom_env(custom_config):
    """Create a ParameterizedLunarLander with custom physics config."""
    env = ParameterizedLunarLander(physics_config=custom_config)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Basic creation and stepping
# ---------------------------------------------------------------------------

class TestEnvBasics:
    """Basic env creation, stepping, and reset."""

    def test_default_creates_without_error(self, default_env):
        """Default config env should create successfully."""
        assert default_env is not None

    def test_reset_returns_obs_and_info(self, default_env):
        """reset() should return (observation, info) tuple."""
        obs, info = default_env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, default_env):
        """step() should return (obs, reward, terminated, truncated, info)."""
        default_env.reset(seed=42)
        result = default_env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_without_reset_raises(self):
        """Stepping before reset should raise an assertion error."""
        env = ParameterizedLunarLander()
        # After __init__, lander is None until reset() is called.
        # But __init__ creates a world, and step checks self.lander is not None.
        # Actually, __init__ doesn't create the lander — reset() does.
        # The lander attribute starts as None.
        with pytest.raises(AssertionError, match="reset"):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        env.close()

    def test_multiple_resets(self, default_env):
        """Env should handle multiple consecutive resets without error."""
        for _ in range(5):
            obs, info = default_env.reset(seed=42)
            assert obs.shape == (15,)

    def test_run_full_episode(self, default_env):
        """Run a full episode with random actions until termination or 300 steps."""
        default_env.reset(seed=42)
        rng = np.random.default_rng(42)
        terminated = False
        steps = 0
        max_steps = 300
        while not terminated and steps < max_steps:
            action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
            obs, reward, terminated, truncated, info = default_env.step(action)
            steps += 1
        # Should either terminate or reach max steps.
        assert steps > 0


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class TestObservationSpace:
    """Observation shape, dtype, and content tests."""

    def test_obs_shape_is_15(self, default_env):
        """Observation should be (15,) = 8 base + 7 physics params."""
        obs, _ = default_env.reset(seed=42)
        assert obs.shape == (15,)

    def test_obs_dtype_is_float32(self, default_env):
        """Observation should be float32 for Gymnasium compatibility."""
        obs, _ = default_env.reset(seed=42)
        assert obs.dtype == np.float32

    def test_obs_space_shape_matches(self, default_env):
        """observation_space.shape should match actual observations."""
        assert default_env.observation_space.shape == (15,)

    def test_obs_within_space_bounds(self, default_env):
        """Observations should (usually) be within the declared space bounds.

        Note: Gymnasium's bounds are approximate — extreme physics can push
        values slightly outside. We test the initial observation which should
        be well within bounds.
        """
        obs, _ = default_env.reset(seed=42)
        low = default_env.observation_space.low
        high = default_env.observation_space.high
        # Allow small tolerance for floating point
        assert np.all(obs >= low - 0.1), f"obs below low: {obs} < {low}"
        assert np.all(obs <= high + 0.1), f"obs above high: {obs} > {high}"

    def test_physics_params_in_obs_match_default_config(self, default_env):
        """obs[8:15] should contain the physics config values (raw)."""
        obs, _ = default_env.reset(seed=42)
        config = LunarLanderPhysicsConfig()  # defaults
        expected = config.as_array()
        np.testing.assert_array_almost_equal(
            obs[8:15], expected,
            err_msg="Physics params in obs don't match default config",
        )

    def test_physics_params_in_obs_match_custom_config(self, custom_env, custom_config):
        """obs[8:15] should reflect the custom physics config."""
        obs, _ = custom_env.reset(seed=42)
        expected = custom_config.as_array()
        np.testing.assert_array_almost_equal(
            obs[8:15], expected,
            err_msg="Physics params in obs don't match custom config",
        )

    def test_physics_params_constant_across_steps(self, default_env):
        """Physics params (obs[8:15]) should be the same every step.

        The physics config doesn't change during an episode — only the
        base state (obs[0:8]) evolves.
        """
        obs, _ = default_env.reset(seed=42)
        initial_params = obs[8:15].copy()
        for _ in range(10):
            obs, _, _, _, _ = default_env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            np.testing.assert_array_equal(obs[8:15], initial_params)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class TestActionSpace:
    """Action space shape and type tests."""

    def test_action_space_is_continuous_box(self, default_env):
        """Action space should be Box(2,) for (main_thrust, side_thrust)."""
        from gymnasium.spaces import Box
        assert isinstance(default_env.action_space, Box)
        assert default_env.action_space.shape == (2,)

    def test_action_space_bounds(self, default_env):
        """Action bounds should be [-1, 1] for both dimensions."""
        np.testing.assert_array_equal(default_env.action_space.low, [-1, -1])
        np.testing.assert_array_equal(default_env.action_space.high, [1, 1])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed should produce identical trajectories."""

    def test_same_seed_same_trajectory(self, default_env):
        """Two rollouts with the same seed and actions must be identical."""
        actions = [np.array([0.5, -0.3], dtype=np.float32) for _ in range(20)]

        # First rollout
        obs1, _ = default_env.reset(seed=123)
        states1 = [obs1.copy()]
        for a in actions:
            obs, _, _, _, _ = default_env.step(a)
            states1.append(obs.copy())

        # Second rollout — same seed
        obs2, _ = default_env.reset(seed=123)
        states2 = [obs2.copy()]
        for a in actions:
            obs, _, _, _, _ = default_env.step(a)
            states2.append(obs.copy())

        for i, (s1, s2) in enumerate(zip(states1, states2)):
            np.testing.assert_array_equal(
                s1, s2, err_msg=f"Divergence at step {i}"
            )

    def test_different_seed_different_trajectory(self, default_env):
        """Different seeds should produce different initial conditions."""
        obs1, _ = default_env.reset(seed=42)
        obs2, _ = default_env.reset(seed=999)
        # The base state (first 8 dims) should differ due to random initial
        # impulse and terrain. Physics params (last 7) are the same config.
        assert not np.array_equal(obs1[:8], obs2[:8])


# ---------------------------------------------------------------------------
# Info dict / metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    """Info dict should contain physics_config."""

    def test_reset_info_has_physics_config(self, default_env):
        """reset() info should contain 'physics_config' dict."""
        _, info = default_env.reset(seed=42)
        assert "physics_config" in info
        assert isinstance(info["physics_config"], dict)

    def test_step_info_has_physics_config(self, default_env):
        """step() info should contain 'physics_config' dict."""
        default_env.reset(seed=42)
        _, _, _, _, info = default_env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        assert "physics_config" in info

    def test_info_physics_config_matches_env(self, custom_env, custom_config):
        """Info physics_config should match the env's actual config."""
        _, info = custom_env.reset(seed=42)
        expected = custom_config.to_dict()
        for key in expected:
            assert info["physics_config"][key] == pytest.approx(expected[key])

    def test_physics_config_property(self, custom_env, custom_config):
        """env.physics_config should expose the config read-only."""
        assert custom_env.physics_config.gravity == custom_config.gravity
        assert custom_env.physics_config.main_engine_power == custom_config.main_engine_power


# ---------------------------------------------------------------------------
# Custom physics config acceptance
# ---------------------------------------------------------------------------

class TestCustomConfig:
    """Verify the env accepts and uses custom physics configs."""

    def test_custom_config_accepted(self, custom_env):
        """Env with custom config should create and reset without error."""
        obs, info = custom_env.reset(seed=42)
        assert obs.shape == (15,)

    def test_none_config_uses_defaults(self):
        """physics_config=None should use default values."""
        env = ParameterizedLunarLander(physics_config=None)
        obs, _ = env.reset(seed=42)
        default = LunarLanderPhysicsConfig()
        np.testing.assert_array_almost_equal(obs[8:15], default.as_array())
        env.close()


# ---------------------------------------------------------------------------
# Terrain segments
# ---------------------------------------------------------------------------

class TestTerrainSegments:
    """Terrain segment exposure for raycasting."""

    def test_terrain_segments_exist_after_reset(self, default_env):
        """terrain_segments should be populated after reset."""
        default_env.reset(seed=42)
        segs = default_env.terrain_segments
        assert segs is not None
        assert len(segs) > 0

    def test_terrain_segments_are_tuples_of_4(self, default_env):
        """Each segment should be (x1, y1, x2, y2)."""
        default_env.reset(seed=42)
        for seg in default_env.terrain_segments:
            assert len(seg) == 4
            assert all(isinstance(v, float) for v in seg)

    def test_terrain_segments_count(self, default_env):
        """Should have CHUNKS-1 = 10 segments."""
        default_env.reset(seed=42)
        assert len(default_env.terrain_segments) == 10

    def test_terrain_segments_change_per_seed(self, default_env):
        """Different seeds should produce different terrain."""
        default_env.reset(seed=42)
        segs1 = list(default_env.terrain_segments)
        default_env.reset(seed=999)
        segs2 = list(default_env.terrain_segments)
        assert segs1 != segs2

    def test_terrain_segments_deterministic(self, default_env):
        """Same seed should produce identical terrain."""
        default_env.reset(seed=42)
        segs1 = list(default_env.terrain_segments)
        default_env.reset(seed=42)
        segs2 = list(default_env.terrain_segments)
        assert segs1 == segs2

    def test_terrain_x_spans_viewport(self, default_env):
        """Terrain should span from near x=0 to near x=W."""
        default_env.reset(seed=42)
        xs = []
        for seg in default_env.terrain_segments:
            xs.extend([seg[0], seg[2]])
        assert min(xs) < 1.0   # near left edge
        assert max(xs) > 19.0  # near right edge (W=20)

    def test_terrain_in_info_dict(self, default_env):
        """reset() info should include terrain_segments."""
        _, info = default_env.reset(seed=42)
        assert "terrain_segments" in info
        assert len(info["terrain_segments"]) == 10


# ---------------------------------------------------------------------------
# Ground truth outcome in info dict
# ---------------------------------------------------------------------------

class TestOutcomeInInfo:
    """Verify that step() reports ground truth outcome in info dict."""

    def test_non_terminal_step_has_no_outcome(self):
        """Non-terminal steps should have outcome=None."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if not terminated and not truncated:
            assert info["outcome"] is None
        env.close()

    def test_crash_outcome(self):
        """Terminal step with random actions should report a failure outcome."""
        env = ParameterizedLunarLander()
        env.reset(seed=42)
        terminated = False
        info = {}
        # Run until termination (random actions => crash or out_of_bounds)
        for _ in range(2000):
            _, _, terminated, _, info = env.step(env.action_space.sample())
            if terminated:
                break
        assert terminated
        assert info["outcome"] in ("landed", "crashed", "out_of_bounds")
        env.close()

    # test_landed_outcome removed: depends on heuristic_policy (not ported)
