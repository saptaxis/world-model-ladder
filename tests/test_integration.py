"""Integration smoke test: train a tiny model, run eval, check outputs."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.loader import EpisodeDataset
from data.normalization import compute_norm_stats
from models.factory import build_model
from training.loop import train_epoch, validate
from evaluation.metrics.core import per_dim_mse, horizon_error_curve
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import RunConfig, generate_run_name


def test_train_eval_roundtrip(episode_dir, tmp_path):
    """End-to-end: train MLP 2 epochs, checkpoint, load, eval."""
    config = RunConfig(
        arch="mlp", arch_params={"hidden_dims": [32]},
        prediction="delta", training_mode="single_step",
        data_mix="policy", data_path=str(episode_dir),
        state_dim=8, action_dim=2,
        lr=1e-3, batch_size=16, epochs=2,
        val_fraction=0.2, run_dir=str(tmp_path),
    )

    # Build data
    train_ds = EpisodeDataset(config.data_path, state_dim=8, mode="single_step",
                              split="train", val_fraction=0.2)
    val_ds = EpisodeDataset(config.data_path, state_dim=8, mode="single_step",
                            split="val", val_fraction=0.2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    norm_stats = compute_norm_stats(train_ds.episode_dicts())
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train 2 epochs
    for epoch in range(2):
        train_epoch(model, train_loader, optimizer, norm_stats,
                    training_mode="single_step")
    val_metrics = validate(model, val_loader, norm_stats,
                           training_mode="single_step")

    # Checkpoint
    ckpt_path = tmp_path / "best.pt"
    save_checkpoint(ckpt_path, model, optimizer, norm_stats, config, epoch=1,
                    metrics=val_metrics)
    assert ckpt_path.exists()

    # Load and verify
    ckpt = load_checkpoint(ckpt_path)
    model2 = build_model(ckpt["config"])
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()

    # Eval A: per-dim MSE
    from data.normalization import NormStats
    ns = NormStats.from_dict(ckpt["norm_stats"])
    pdm = per_dim_mse(model2, val_loader, ns)
    assert pdm.shape == (8,)
    assert (pdm >= 0).all()
    assert pdm.mean() < 100  # sanity: not exploded

    # Eval B: horizon curves (short horizons since episodes are 50 steps)
    curves = horizon_error_curve(model2, val_ds, ns, horizons=[1, 5, 10],
                                 n_rollouts=5)
    assert 1 in curves
    assert curves[1].shape == (8,)


def test_multi_step_train_roundtrip(episode_dir, tmp_path):
    """End-to-end with multi-step training."""
    config = RunConfig(
        arch="mlp", arch_params={"hidden_dims": [32]},
        prediction="delta", training_mode="multi_step",
        rollout_k=5, data_mix="policy", data_path=str(episode_dir),
        state_dim=8, action_dim=2,
        lr=1e-3, batch_size=8, epochs=2, seq_len=10,
        val_fraction=0.2, run_dir=str(tmp_path),
    )

    train_ds = EpisodeDataset(config.data_path, state_dim=8, mode="sequence",
                              seq_len=10, split="train", val_fraction=0.2)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)

    norm_stats = compute_norm_stats(train_ds.episode_dicts())
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(2):
        metrics = train_epoch(model, train_loader, optimizer, norm_stats,
                              training_mode="multi_step", rollout_k=5)
    assert metrics["train_loss"] > 0
    assert metrics["train_loss"] < 1000  # not exploded


def test_train_script_smoke(episode_dir, tmp_path):
    """Smoke test: run train.py as a subprocess with tiny config."""
    import subprocess
    import sys

    config = RunConfig(
        arch="linear", arch_params={},
        prediction="delta", training_mode="single_step",
        data_mix="policy", data_path=str(episode_dir),
        state_dim=8, action_dim=2,
        lr=1e-3, batch_size=16, epochs=2,
        val_fraction=0.2, run_dir=str(tmp_path),
    )
    cfg_path = tmp_path / "smoke.yaml"
    config.save(cfg_path)

    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    assert (tmp_path / "linear-delta-single_step_k1-policy" / "best.pt").exists()


def test_eval_script_smoke(episode_dir, tmp_path):
    """Smoke test: train then eval as subprocesses."""
    import subprocess
    import sys

    config = RunConfig(
        arch="linear", arch_params={},
        prediction="delta", training_mode="single_step",
        data_mix="policy", data_path=str(episode_dir),
        state_dim=8, action_dim=2,
        lr=1e-3, batch_size=16, epochs=2,
        val_fraction=0.2, run_dir=str(tmp_path),
    )
    cfg_path = tmp_path / "smoke.yaml"
    config.save(cfg_path)

    # Train
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=60, check=True,
    )

    ckpt = tmp_path / "linear-delta-single_step_k1-policy" / "best.pt"
    assert ckpt.exists()

    # Eval
    result = subprocess.run(
        [sys.executable, "scripts/eval.py", "--checkpoint", str(ckpt)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"eval.py failed:\n{result.stderr}"
    metrics_json = ckpt.parent / "eval" / "metrics.json"
    assert metrics_json.exists()
