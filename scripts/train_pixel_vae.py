#!/usr/bin/env python3
"""Train a PixelVAE (Phase 1 of pixel world model).

Usage:
    python scripts/train_pixel_vae.py \
        --data-path /path/to/episodes \
        --run-dir runs/pixel-vae \
        --epochs 50 --batch-size 128
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.pixel_dataset import PixelFrameDataset
from models.pixel_vae import PixelVAE
from training.callbacks import (
    CallbackContext,
    CheckpointCallback,
    GradNormCallback,
    NaNDetectionCallback,
    ProgressCallback,
)
from training.pixel_callbacks import PixelVAEValidationCallback, ReconGridCallback
from training.pixel_loop import pixel_vae_train_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train PixelVAE")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--run-dir", type=str, default="runs/pixel-vae")
    p.add_argument("--frame-size", type=int, default=84)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--beta", type=float, default=0.0001)
    p.add_argument("--fg-weight", type=float, default=1.0,
                   help="Foreground pixel weight in recon loss. >1 upweights lander/flames vs sky/terrain.")
    p.add_argument("--state-dim", type=int, default=0,
                   help="Aux state prediction head dim. 6 = (x,y,vx,vy,angle,ang_vel). 0 = disabled.")
    p.add_argument("--state-weight", type=float, default=1.0,
                   help="Weight for auxiliary state prediction loss (only if --state-dim > 0).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers for batch prefetching")
    p.add_argument("--load-workers", type=int, default=8,
                   help="Parallel workers for initial npz loading")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Dir to cache preprocessed frames (.npy). Instant load on reruns.")
    p.add_argument("--lr-patience", type=int, default=0,
                   help="ReduceLROnPlateau patience (0 = no scheduler)")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="LR reduction factor when plateau detected")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum LR for scheduler")
    p.add_argument("--grayscale", action="store_true", default=True)
    p.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    # Directories — run_dir IS the VAE run dir (not a parent)
    vae_dir = Path(args.run_dir)
    vae_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(vae_dir)

    # Data — use cache if available for instant loading
    cache_train = None
    cache_val = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        gs_tag = "gray" if args.grayscale else "rgb"
        state_tag = f"_s{args.state_dim}" if args.state_dim > 0 else ""
        cache_train = cache_dir / f"frames_train_{args.frame_size}_{gs_tag}{state_tag}.npz"
        cache_val = cache_dir / f"frames_val_{args.frame_size}_{gs_tag}{state_tag}.npz"

    print(f"Loading data from {args.data_path} ...")
    train_ds = PixelFrameDataset(
        args.data_path, frame_size=args.frame_size,
        grayscale=args.grayscale, split="train",
        n_workers=args.load_workers, cache_path=cache_train,
        state_dim=args.state_dim,
    )
    val_ds = PixelFrameDataset(
        args.data_path, frame_size=args.frame_size,
        grayscale=args.grayscale, split="val",
        n_workers=args.load_workers, cache_path=cache_val,
        state_dim=args.state_dim,
    )
    print(f"  Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    grayscale_channels = 1 if args.grayscale else 3
    in_ch = args.in_channels if args.in_channels != 1 else grayscale_channels
    vae = PixelVAE(
        in_channels=in_ch,
        latent_dim=args.latent_dim,
        frame_size=args.frame_size,
        channels=args.channels,
        beta=args.beta,
        state_dim=args.state_dim,
    ).to(args.device)

    param_count = sum(p.numel() for p in vae.parameters())
    print(f"PixelVAE: {param_count:,} parameters")

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min, verbose=True,
        )
        print(f"LR scheduler: ReduceLROnPlateau(patience={args.lr_patience}, "
              f"factor={args.lr_factor}, min_lr={args.lr_min})")

    # TensorBoard
    writer = SummaryWriter(log_dir=str(vae_dir / "tb"))

    # Config dict — saved to checkpoint AND config.json for reproducibility
    config = {
        "in_channels": in_ch,
        "latent_dim": args.latent_dim,
        "frame_size": args.frame_size,
        "channels": args.channels,
        "beta": args.beta,
        "fg_weight": args.fg_weight,
        "state_dim": args.state_dim,
        "state_weight": args.state_weight,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "grad_clip": args.grad_clip,
        "data_path": args.data_path,
        "grayscale": args.grayscale,
    }
    import json
    with open(vae_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {vae_dir / 'config.json'}")

    # Callbacks
    # Get a sample batch for ReconGridCallback
    sample_batch = next(iter(val_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]

    callbacks = [
        PixelVAEValidationCallback(
            val_loader=val_loader, beta=args.beta,
            fg_weight=args.fg_weight,
            state_weight=args.state_weight if args.state_dim > 0 else 0.0,
            every_n_steps=args.val_every, patience=args.patience,
            checkpoint_dir=ckpt_dir,
        ),
        CheckpointCallback(checkpoint_dir=ckpt_dir, every_n_steps=args.ckpt_every),
        ReconGridCallback(sample_batch=sample_batch, every_n_steps=args.val_every),
        GradNormCallback(every_n_steps=50),
        NaNDetectionCallback(),
        ProgressCallback(every_n_steps=100, total_epochs=args.epochs),
    ]

    ctx = CallbackContext(
        model=vae, optimizer=optimizer, writer=writer,
        global_step=0, epoch=0, run_dir=str(vae_dir),
        device=args.device,
        extras={"config": config},
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
            "model_state_dict": vae.state_dict(),
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

    print(f"\nTraining VAE for {args.epochs} epochs on {args.device}")
    for epoch in range(args.epochs):
        ctx.epoch = epoch
        result = pixel_vae_train_epoch(
            vae, train_loader, optimizer,
            beta=args.beta, fg_weight=args.fg_weight,
            state_weight=args.state_weight if args.state_dim > 0 else 0.0,
            device=args.device,
            max_grad_norm=args.grad_clip,
            ctx=ctx, callbacks=callbacks,
        )

        state_str = f" state={result['state_loss']:.6f}" if args.state_dim > 0 else ""
        print(f"Epoch {epoch}: train_loss={result['train_loss']:.6f} "
              f"recon={result['recon_loss']:.6f} kl={result['kl_loss']:.6f}{state_str}")

        if ctx.writer:
            ctx.writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], ctx.global_step)

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
    print(f"\nVAE training complete. Best checkpoint: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
