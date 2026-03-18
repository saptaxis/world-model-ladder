#!/usr/bin/env python3
"""Train LatentDynamicsModel (Phase 2 of pixel world model).

Loads a frozen VAE from checkpoint, trains GRU dynamics in latent space.

Usage:
    python scripts/train_pixel_dynamics.py \
        --vae-checkpoint path/to/vae/best.pt \
        --data-path /path/to/episodes \
        --run-dir runs/pixel-wm
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.pixel_dataset import PixelEpisodeDataset
from models.pixel_vae import PixelVAE
from models.pixel_dynamics import LatentDynamicsModel
from training.callbacks import (
    CallbackContext,
    CheckpointCallback,
    GradNormCallback,
    NaNDetectionCallback,
    ProgressCallback,
)
from training.pixel_callbacks import PixelDynamicsValidationCallback, DreamGridCallback
from training.pixel_loop import pixel_dynamics_train_epoch


def load_vae(checkpoint_path: str, device: str) -> PixelVAE:
    """Load VAE from checkpoint, extracting config to recreate architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vae = PixelVAE(
        in_channels=cfg["in_channels"],
        latent_dim=cfg["latent_dim"],
        frame_size=cfg["frame_size"],
        channels=cfg.get("channels", [32, 64, 128, 256]),
        state_dim=cfg.get("state_dim", 0),
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def get_sampling_prob(epoch: int, total_epochs: int,
                      start: float, end: float, warmup_frac: float) -> float:
    """Linear annealing of scheduled sampling probability."""
    warmup_epochs = int(total_epochs * warmup_frac)
    if epoch < warmup_epochs:
        return start
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return start + (end - start) * min(progress, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Train LatentDynamicsModel")
    p.add_argument("--vae-checkpoint", type=str, required=True)
    p.add_argument("--data-path", type=str, nargs="+", required=True,
                   help="One or more directories with .npz episode files")
    p.add_argument("--run-dir", type=str, default="runs/pixel-wm")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--sampling-start", type=float, default=0.0)
    p.add_argument("--sampling-end", type=float, default=0.5)
    p.add_argument("--sampling-warmup-frac", type=float, default=0.5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers for batch prefetching")
    p.add_argument("--load-workers", type=int, default=8,
                   help="Parallel workers for initial npz loading")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Dir to cache preprocessed episodes. Instant load on reruns.")
    p.add_argument("--lr-patience", type=int, default=0,
                   help="ReduceLROnPlateau patience (0 = no scheduler)")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="LR reduction factor when plateau detected")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum LR for scheduler")
    return p.parse_args()


def main():
    args = parse_args()

    # Load frozen VAE
    print(f"Loading VAE from {args.vae_checkpoint} ...")
    vae = load_vae(args.vae_checkpoint, args.device)
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=args.device, weights_only=False)
    vae_cfg = vae_ckpt["config"]
    frame_size = vae_cfg["frame_size"]
    latent_dim = vae_cfg["latent_dim"]
    grayscale = vae_cfg["in_channels"] == 1
    print(f"  VAE: latent_dim={latent_dim}, frame_size={frame_size}")

    # Directories — run_dir IS the dynamics run dir (not a parent)
    dyn_dir = Path(args.run_dir)
    dyn_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(dyn_dir)

    # Dump full config so we know exactly what this run was trained with
    import json
    run_config = {
        "vae_checkpoint": str(Path(args.vae_checkpoint).resolve()),
        "vae_config": vae_cfg,
        "data_path": args.data_path,
        "hidden_size": args.hidden_size,
        "action_dim": args.action_dim,
        "seq_len": args.seq_len,
        "frame_stack": args.frame_stack,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "sampling_start": args.sampling_start,
        "sampling_end": args.sampling_end,
        "sampling_warmup_frac": args.sampling_warmup_frac,
        "device": args.device,
    }
    with open(dyn_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config saved to {dyn_dir / 'config.json'}")

    # Data
    print(f"Loading data from {args.data_path} ...")
    cache_train = None
    cache_val = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        gs_tag = "gray" if grayscale else "rgb"
        cache_train = cache_dir / f"episodes_train_{frame_size}_{gs_tag}.npz"
        cache_val = cache_dir / f"episodes_val_{frame_size}_{gs_tag}.npz"

    train_ds = PixelEpisodeDataset(
        args.data_path, frame_size=frame_size, grayscale=grayscale,
        seq_len=args.seq_len, frame_stack=args.frame_stack, split="train",
        n_workers=args.load_workers, cache_path=cache_train,
    )
    val_ds = PixelEpisodeDataset(
        args.data_path, frame_size=frame_size, grayscale=grayscale,
        seq_len=args.seq_len, frame_stack=args.frame_stack, split="val",
        n_workers=args.load_workers, cache_path=cache_val,
    )
    print(f"  Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    dynamics = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
    ).to(args.device)

    param_count = sum(p.numel() for p in dynamics.parameters())
    print(f"LatentDynamicsModel: {param_count:,} parameters")

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=args.lr)

    # LR scheduler (optional — only if --lr-patience > 0)
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min, verbose=True,
        )
        print(f"LR scheduler: ReduceLROnPlateau(patience={args.lr_patience}, "
              f"factor={args.lr_factor}, min_lr={args.lr_min})")

    writer = SummaryWriter(log_dir=str(dyn_dir / "tb"))

    config = {
        "hidden_size": args.hidden_size,
        "action_dim": args.action_dim,
        "seq_len": args.seq_len,
        "frame_stack": args.frame_stack,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "vae_checkpoint": args.vae_checkpoint,
    }

    # Get a sample for DreamGridCallback
    sample_frames, sample_actions = next(iter(val_loader))
    # Use first episode, first frame as seed, first few actions
    sample_seed = sample_frames[0:1, 0]  # (1, C, H, W)
    sample_act = sample_actions[0:1, :min(10, sample_actions.size(1))]  # (1, T, action_dim)

    callbacks = [
        PixelDynamicsValidationCallback(
            val_loader=val_loader, vae=vae,
            every_n_steps=args.val_every, patience=args.patience,
            checkpoint_dir=ckpt_dir,
        ),
        CheckpointCallback(checkpoint_dir=ckpt_dir, every_n_steps=args.ckpt_every),
        DreamGridCallback(vae=vae, sample_frames=sample_seed,
                          sample_actions=sample_act, every_n_steps=args.val_every * 2),
        GradNormCallback(every_n_steps=50),
        NaNDetectionCallback(),
        ProgressCallback(every_n_steps=100, total_epochs=args.epochs),
    ]

    ctx = CallbackContext(
        model=dynamics, optimizer=optimizer, writer=writer,
        global_step=0, epoch=0, run_dir=str(dyn_dir),
        device=args.device,
        extras={"config": config, "vae_checkpoint": args.vae_checkpoint},
    )

    # SIGINT handler
    interrupted = False
    def handle_sigint(sig, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True
        print("\nInterrupted! Saving checkpoint and exiting...")
        torch.save({
            "model_state_dict": dynamics.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": ctx.epoch,
            "global_step": ctx.global_step,
            "config": config,
        }, Path(ckpt_dir) / "interrupted.pt")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # Train
    for cb in callbacks:
        cb.on_train_start(ctx)

    print(f"\nTraining dynamics for {args.epochs} epochs on {args.device}")
    for epoch in range(args.epochs):
        ctx.epoch = epoch

        sampling_prob = get_sampling_prob(
            epoch, args.epochs,
            args.sampling_start, args.sampling_end,
            args.sampling_warmup_frac,
        )

        result = pixel_dynamics_train_epoch(
            dynamics, vae, train_loader, optimizer,
            sampling_prob=sampling_prob,
            device=args.device,
            max_grad_norm=args.grad_clip,
            ctx=ctx, callbacks=callbacks,
        )

        print(f"Epoch {epoch}: train_loss={result['train_loss']:.6f} "
              f"sampling_prob={sampling_prob:.3f}")

        if ctx.writer:
            ctx.writer.add_scalar("train/sampling_prob", sampling_prob, ctx.global_step)
            ctx.writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], ctx.global_step)

        # Step LR scheduler on val loss
        if scheduler is not None and "val_loss" in ctx.extras:
            scheduler.step(ctx.extras["val_loss"])

        for cb in callbacks:
            if cb.on_epoch_end(ctx) is False:
                print("Early stopping triggered.")
                break

        if result.get("stop_requested"):
            print("Stop requested by callback.")
            break

    for cb in callbacks:
        cb.on_train_end(ctx)

    writer.close()
    print(f"\nDynamics training complete. Best checkpoint: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
