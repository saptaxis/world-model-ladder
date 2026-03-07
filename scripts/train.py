#!/usr/bin/env python3
"""Train a world model.

Usage:
    python scripts/train.py --config configs/examples/mlp-single-step.yaml
    python scripts/train.py --config configs/examples/mlp-single-step.yaml --rollout_k 10 --suffix k10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.loader import EpisodeDataset
from data.normalization import compute_norm_stats
from models.factory import build_model
from training.loop import train_epoch, validate
from training.scheduling import curriculum_schedule, sampling_schedule
from utils.checkpoint import save_checkpoint
from utils.config import RunConfig, load_config, generate_run_name
from utils.logging import TrainLogger, DIM_NAMES_8D


def parse_args():
    parser = argparse.ArgumentParser(description="Train a world model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    # Allow any RunConfig field as an override
    parser.add_argument("--rollout_k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Build overrides dict from non-None CLI args
    overrides = {}
    for field in ["rollout_k", "lr", "batch_size", "epochs", "suffix", "kl_weight"]:
        val = getattr(args, field)
        if val is not None:
            overrides[field] = val

    config = load_config(args.config, overrides=overrides if overrides else None)
    run_name = generate_run_name(config)
    run_dir = Path(config.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Output: {run_dir}")

    # Data
    data_mode = "single_step" if config.training_mode == "single_step" else "sequence"
    seq_len = config.seq_len if data_mode == "sequence" else None

    train_ds = EpisodeDataset(config.data_path, state_dim=config.state_dim,
                              mode=data_mode, seq_len=seq_len, split="train",
                              val_fraction=config.val_fraction)
    val_ds = EpisodeDataset(config.data_path, state_dim=config.state_dim,
                            mode=data_mode, seq_len=seq_len, split="val",
                            val_fraction=config.val_fraction)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    print(f"Train: {train_ds.n_episodes} episodes, {len(train_ds)} samples")
    print(f"Val: {val_ds.n_episodes} episodes, {len(val_ds)} samples")

    # Normalization stats from training data
    norm_stats = compute_norm_stats(train_ds.episode_dicts())

    # Model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.arch} ({n_params:,} params)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Logging
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    logger = TrainLogger(writer)

    # Training
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        # Compute schedule values
        current_k = config.rollout_k
        current_sampling_prob = 0.0
        if config.curriculum:
            current_k = curriculum_schedule(epoch, config.epochs,
                                            k_min=1, k_max=config.rollout_k)
        if config.training_mode == "scheduled_sampling":
            current_sampling_prob = sampling_schedule(
                epoch, config.epochs,
                start=config.sampling_start, end=config.sampling_end)

        train_metrics = train_epoch(
            model, train_loader, optimizer, norm_stats,
            training_mode=config.training_mode, rollout_k=current_k,
            device=device, sampling_prob=current_sampling_prob,
            kl_weight=config.kl_weight,
        )
        val_metrics = validate(
            model, val_loader, norm_stats,
            training_mode=config.training_mode, rollout_k=current_k,
            device=device, sampling_prob=current_sampling_prob,
            kl_weight=config.kl_weight,
        )

        logger.log_scalar("train/loss", train_metrics["train_loss"], epoch)
        logger.log_scalar("val/loss", val_metrics["val_loss"], epoch)

        # Checkpoint
        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, norm_stats,
                            config, epoch, val_metrics)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, optimizer,
                            norm_stats, config, epoch, val_metrics)
            print(f"  Epoch {epoch:3d}  train={train_metrics['train_loss']:.6f}  "
                  f"val={val_metrics['val_loss']:.6f}  {'*' if is_best else ''}")

    writer.close()
    print(f"Done. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints in: {run_dir}")


if __name__ == "__main__":
    main()
