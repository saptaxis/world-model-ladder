"""Smoke-test train_pixel_dynamics script configuration."""


def test_factored_dyn_in_valid_combos():
    """factored-dyn model type is registered in VALID_COMBOS."""
    with open("scripts/train_pixel_dynamics.py") as f:
        content = f.read()
    assert '("factored-dyn", "latent_mse")' in content
    assert '("factored-dyn", "multi_step_latent")' in content


def test_factored_dyn_in_argparse_choices():
    """factored-dyn appears in --model-type choices."""
    with open("scripts/train_pixel_dynamics.py") as f:
        content = f.read()
    assert '"factored-dyn"' in content
