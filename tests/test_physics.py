"""Tests for physics parameterization — verify that changing parameters
actually produces different physical behavior.

These tests don't check exact values (that's calibration's job) — they
check that the parameter has a measurable, directionally-correct effect
on the simulation. E.g., stronger gravity should make the lander fall
faster, more thrust should make it rise faster, etc.

Methodology: Create two envs with identical seeds but different physics
configs (varying one parameter at a time). Run the same actions for N
steps. Compare a relevant quantity (position, velocity, etc.) to verify
the parameter had the expected effect.
"""

import numpy as np
import pytest

from envs.lunar_lander import ParameterizedLunarLander, LunarLanderPhysicsConfig


def run_steps(env, actions, seed=42):
    """Run a sequence of actions and return all observations.

    Args:
        env: ParameterizedLunarLander instance.
        actions: List of (2,) action arrays, or a single action to repeat.
        seed: RNG seed for reset.

    Returns:
        List of observations (including the initial obs from reset).
    """
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]
    if isinstance(actions, np.ndarray) and actions.ndim == 1:
        # Single action repeated for... we need a count. Use as-is.
        actions = [actions]
    for action in actions:
        obs, _, terminated, _, _ = env.step(action)
        observations.append(obs.copy())
        if terminated:
            break
    return observations


def make_env(config):
    """Shorthand to create env with a config."""
    return ParameterizedLunarLander(physics_config=config)


# No-thrust action (engines off).
NO_THRUST = np.array([0.0, 0.0], dtype=np.float32)
# Full main engine thrust.
FULL_MAIN = np.array([1.0, 0.0], dtype=np.float32)
# Full right side engine.
FULL_SIDE = np.array([0.0, 1.0], dtype=np.float32)

N_STEPS = 50  # Number of steps for comparison tests


class TestGravityEffect:
    """Stronger (more negative) gravity should make the lander fall faster."""

    def test_stronger_gravity_falls_faster(self):
        """With no thrust, stronger gravity should produce lower y position
        after N steps (lander falls further from starting height)."""
        # Weak gravity: lander falls slowly.
        config_weak = LunarLanderPhysicsConfig(
            gravity=-4.0, wind_power=0.0, turbulence_power=0.0
        )
        # Strong gravity: lander falls fast.
        config_strong = LunarLanderPhysicsConfig(
            gravity=-12.0, wind_power=0.0, turbulence_power=0.0
        )

        env_weak = make_env(config_weak)
        env_strong = make_env(config_strong)

        try:
            actions = [NO_THRUST] * N_STEPS
            obs_weak = run_steps(env_weak, actions)
            obs_strong = run_steps(env_strong, actions)

            # Compare y position (obs[1]) at the last available step.
            # Stronger gravity → lower y (more negative in normalized coords).
            # Use min of both trajectory lengths (strong gravity may terminate
            # earlier due to crashing).
            min_len = min(len(obs_weak), len(obs_strong))
            assert min_len > 5, "Both trajectories should run at least 5 steps"

            # At step 5 (early enough that neither has crashed), compare y.
            y_weak = obs_weak[5][1]
            y_strong = obs_strong[5][1]
            assert y_strong < y_weak, (
                f"Stronger gravity should produce lower y: "
                f"strong={y_strong:.3f} vs weak={y_weak:.3f}"
            )
        finally:
            env_weak.close()
            env_strong.close()


class TestMainEnginePowerEffect:
    """Stronger main engine should produce more upward acceleration."""

    def test_more_thrust_rises_faster(self):
        """With full main thrust, higher main_engine_power should produce
        higher y velocity after N steps."""
        config_weak = LunarLanderPhysicsConfig(
            main_engine_power=6.0, wind_power=0.0, turbulence_power=0.0
        )
        config_strong = LunarLanderPhysicsConfig(
            main_engine_power=24.0, wind_power=0.0, turbulence_power=0.0
        )

        env_weak = make_env(config_weak)
        env_strong = make_env(config_strong)

        try:
            actions = [FULL_MAIN] * 20
            obs_weak = run_steps(env_weak, actions)
            obs_strong = run_steps(env_strong, actions)

            # Compare y velocity (obs[3]) after 10 steps.
            # Higher thrust → more positive vy (upward).
            min_len = min(len(obs_weak), len(obs_strong))
            step = min(10, min_len - 1)
            vy_weak = obs_weak[step][3]
            vy_strong = obs_strong[step][3]
            assert vy_strong > vy_weak, (
                f"Stronger thrust should produce higher vy: "
                f"strong={vy_strong:.3f} vs weak={vy_weak:.3f}"
            )
        finally:
            env_weak.close()
            env_strong.close()


class TestLanderDensityEffect:
    """Heavier lander (higher density) should accelerate more slowly."""

    def test_heavier_lander_falls_faster(self):
        """In free fall, a heavier lander should... actually fall at the same
        rate (gravity is independent of mass in free fall). But with thrust,
        heavier lander should accelerate less (F=ma, same F, bigger m).

        Test: Apply full main thrust. Lighter lander should achieve higher
        y velocity (more acceleration for same force).
        """
        config_light = LunarLanderPhysicsConfig(
            lander_density=2.5, wind_power=0.0, turbulence_power=0.0
        )
        config_heavy = LunarLanderPhysicsConfig(
            lander_density=10.0, wind_power=0.0, turbulence_power=0.0
        )

        env_light = make_env(config_light)
        env_heavy = make_env(config_heavy)

        try:
            actions = [FULL_MAIN] * 20
            obs_light = run_steps(env_light, actions)
            obs_heavy = run_steps(env_heavy, actions)

            # After 10 steps of full thrust, lighter lander should have
            # higher y velocity (thrust/mass is larger).
            min_len = min(len(obs_light), len(obs_heavy))
            step = min(10, min_len - 1)
            vy_light = obs_light[step][3]
            vy_heavy = obs_heavy[step][3]
            assert vy_light > vy_heavy, (
                f"Lighter lander should accelerate faster with same thrust: "
                f"light_vy={vy_light:.3f} vs heavy_vy={vy_heavy:.3f}"
            )
        finally:
            env_light.close()
            env_heavy.close()


class TestWindPowerEffect:
    """Wind should cause trajectory divergence from the no-wind baseline."""

    def test_wind_causes_trajectory_divergence(self):
        """With the same seed and actions, wind_power=25 should cause the
        x-position trajectory to diverge from the wind_power=0 baseline.

        Both envs start with identical initial conditions (same seed → same
        terrain, same initial impulse). The ONLY difference is wind force.
        So any divergence in x-position is attributable to wind.

        We can't predict which DIRECTION wind pushes (it's quasi-periodic
        and could oppose the initial impulse), but we can verify the
        trajectories differ.
        """
        config_no_wind = LunarLanderPhysicsConfig(
            wind_power=0.0, turbulence_power=0.0
        )
        config_windy = LunarLanderPhysicsConfig(
            wind_power=25.0, turbulence_power=0.0
        )

        env_calm = make_env(config_no_wind)
        env_windy = make_env(config_windy)

        try:
            actions = [NO_THRUST] * 30
            obs_calm = run_steps(env_calm, actions)
            obs_windy = run_steps(env_windy, actions)

            # Both trajectories start from identical initial conditions
            # (same seed → same RNG state → same initial impulse, terrain).
            # Any difference in x position is caused by wind.
            min_len = min(len(obs_calm), len(obs_windy))
            assert min_len > 10, "Both trajectories should run at least 10 steps"

            # Measure divergence: |x_windy - x_calm| at a late step.
            # This should be nonzero because wind applies force every step.
            step = min(20, min_len - 1)
            x_divergence = abs(obs_windy[step][0] - obs_calm[step][0])

            assert x_divergence > 0.001, (
                f"Wind should cause trajectory divergence from baseline: "
                f"|x_windy - x_calm| = {x_divergence:.6f} at step {step}"
            )
        finally:
            env_calm.close()
            env_windy.close()


class TestAngularDampingEffect:
    """Higher angular damping should reduce angular velocity faster."""

    def test_high_damping_reduces_angular_velocity(self):
        """With angular_damping=5, angular velocity should decay toward zero
        faster than with angular_damping=0 (free rotation)."""
        config_free = LunarLanderPhysicsConfig(
            angular_damping=0.0, wind_power=0.0, turbulence_power=0.0
        )
        config_damped = LunarLanderPhysicsConfig(
            angular_damping=5.0, wind_power=0.0, turbulence_power=0.0
        )

        env_free = make_env(config_free)
        env_damped = make_env(config_damped)

        try:
            # Use side thrust for a few steps to induce rotation,
            # then let it coast with no thrust.
            spin_actions = [FULL_SIDE] * 5
            coast_actions = [NO_THRUST] * 30
            all_actions = spin_actions + coast_actions

            obs_free = run_steps(env_free, all_actions)
            obs_damped = run_steps(env_damped, all_actions)

            # After the spin phase (step 5) and some coasting (step 20),
            # the damped env should have lower |angular_velocity| (obs[5]).
            min_len = min(len(obs_free), len(obs_damped))
            coast_step = min(25, min_len - 1)

            # obs[5] = angular velocity (scaled by 20/FPS)
            ang_vel_free = abs(obs_free[coast_step][5])
            ang_vel_damped = abs(obs_damped[coast_step][5])

            assert ang_vel_damped < ang_vel_free, (
                f"Damped angular velocity should be smaller: "
                f"damped={ang_vel_damped:.3f} vs free={ang_vel_free:.3f}"
            )
        finally:
            env_free.close()
            env_damped.close()


class TestSideEnginePowerEffect:
    """Stronger side engines should produce more lateral acceleration."""

    def test_stronger_side_engine(self):
        """Higher side_engine_power should produce more lateral velocity
        after applying side thrust for N steps."""
        config_weak = LunarLanderPhysicsConfig(
            side_engine_power=0.2, wind_power=0.0, turbulence_power=0.0
        )
        config_strong = LunarLanderPhysicsConfig(
            side_engine_power=1.5, wind_power=0.0, turbulence_power=0.0
        )

        env_weak = make_env(config_weak)
        env_strong = make_env(config_strong)

        try:
            # Apply side thrust for 10 steps, then compare x velocity.
            actions = [FULL_SIDE] * 10
            obs_weak = run_steps(env_weak, actions)
            obs_strong = run_steps(env_strong, actions)

            # Compare absolute x velocity at step 10.
            # Side engines create both lateral force AND torque, so the
            # effect is complex. But stronger engines should produce more
            # angular displacement at minimum.
            min_len = min(len(obs_weak), len(obs_strong))
            step = min(10, min_len - 1)

            # Check angular velocity OR x velocity — side engines primarily
            # affect rotation which then indirectly affects lateral motion.
            ang_vel_weak = abs(obs_weak[step][5])
            ang_vel_strong = abs(obs_strong[step][5])

            assert ang_vel_strong > ang_vel_weak, (
                f"Stronger side engine should produce more angular effect: "
                f"strong={ang_vel_strong:.3f} vs weak={ang_vel_weak:.3f}"
            )
        finally:
            env_weak.close()
            env_strong.close()


class TestExtremeConfigs:
    """Extreme physics configs should not crash the environment."""

    @pytest.mark.parametrize("config_name,config", [
        ("min_gravity_max_thrust", LunarLanderPhysicsConfig(
            gravity=-2.0, main_engine_power=25.0, side_engine_power=1.5,
            lander_density=2.5, angular_damping=5.0,
            wind_power=0.0, turbulence_power=0.0,
        )),
        ("max_gravity_min_thrust", LunarLanderPhysicsConfig(
            gravity=-12.0, main_engine_power=5.0, side_engine_power=0.2,
            lander_density=10.0, angular_damping=0.0,
            wind_power=0.0, turbulence_power=0.0,
        )),
        ("max_wind", LunarLanderPhysicsConfig(
            wind_power=30.0, turbulence_power=5.0,
        )),
        ("all_minimums", LunarLanderPhysicsConfig(
            gravity=-12.0, main_engine_power=5.0, side_engine_power=0.2,
            lander_density=2.5, angular_damping=0.0,
            wind_power=0.0, turbulence_power=0.0,
        )),
        ("all_maximums", LunarLanderPhysicsConfig(
            gravity=-2.0, main_engine_power=25.0, side_engine_power=1.5,
            lander_density=10.0, angular_damping=5.0,
            wind_power=30.0, turbulence_power=5.0,
        )),
    ])
    def test_extreme_config_no_crash(self, config_name, config):
        """Extreme configs should create, reset, and step without errors.

        Some configs will be physically uncontrollable (e.g., thrust-to-weight
        ratio < 1), but the simulation itself should not crash.
        """
        env = make_env(config)
        try:
            obs, info = env.reset(seed=42)
            assert obs.shape == (15,)
            rng = np.random.default_rng(42)
            for _ in range(100):
                action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
                obs, _, terminated, _, _ = env.step(action)
                if terminated:
                    # Reset and keep going — we want 100 steps total.
                    obs, _ = env.reset(seed=None)
        finally:
            env.close()
