"""Tests for plotting utilities."""
from utils.plotting import plot_horizon_curve, plot_per_dim_bars


def test_plot_horizon_curve(tmp_path):
    horizon_data = {1: 0.001, 5: 0.01, 10: 0.1, 20: 0.5, 50: 2.0}
    out = tmp_path / "horizon.png"
    plot_horizon_curve(horizon_data, str(out), title="Test Model")
    assert out.exists()


def test_plot_per_dim_bars(tmp_path):
    per_dim = {"x": 0.001, "y": 0.002, "vx": 0.01, "vy": 0.005,
               "angle": 0.003, "angular_vel": 0.008,
               "left_leg": 0.0001, "right_leg": 0.0002}
    out = tmp_path / "per_dim.png"
    plot_per_dim_bars(per_dim, str(out), title="Test Model")
    assert out.exists()


def test_plot_horizon_curve_log_scale(tmp_path):
    horizon_data = {1: 0.001, 5: 0.1, 10: 10.0, 20: 1000.0}
    out = tmp_path / "horizon_log.png"
    plot_horizon_curve(horizon_data, str(out), log_scale=True)
    assert out.exists()
