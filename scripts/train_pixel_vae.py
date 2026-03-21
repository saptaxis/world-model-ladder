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
    p.add_argument("--data-path", type=str, nargs="+", required=True,
                   help="One or more directories with .npz episode files")
    p.add_argument("--run-dir", type=str, default="runs/pixel-vae")
    # 84x84 matches the standard Atari/control benchmark resolution and is
    # small enough that a 4-layer conv encoder reaches a 4x4 or 6x6 spatial
    # bottleneck, keeping the VAE compact.
    p.add_argument("--frame-size", type=int, default=84)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--in-channels", type=int, default=1)
    # 4-layer encoder: 32→64→128→256 channels.  Each layer halves spatial
    # resolution, so 84→42→21→10→5 (approx) before flattening to latent.
    p.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    # beta=0.0001 is intentionally low — we prioritise sharp reconstructions
    # over a smooth latent space. Dynamics training regularises the latent
    # space downstream, so heavy KL pressure here would hurt pixel quality
    # without benefit.
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
    # patience=10 val checks without improvement triggers early stop, which
    # prevents overfitting the decoder to training-set textures.
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # num-workers controls DataLoader prefetch parallelism (batch-level);
    # load-workers controls the initial npz→numpy bulk load (episode-level).
    # They serve different phases, so both are independently tunable.
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
    # Default to grayscale because LunarLander's colour information is
    # mostly decorative (sky gradient, terrain shade) — grayscale preserves
    # the structurally important edges (lander, legs, flames) at 1/3 the
    # channel cost.
    p.add_argument("--grayscale", action="store_true", default=True)
    p.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pt) to resume from. Restores model weights, "
                        "optimizer state, epoch, and global_step.")
    return p.parse_args()


def main():
    args = parse_args()

    # Directories — run_dir IS the VAE run dir (not a parent)
    vae_dir = Path(args.run_dir)
    vae_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(vae_dir)

    # --- Data loading ---
    # Cache paths encode frame_size, grayscale, and state_dim so that
    # different preprocessing configs don't collide in the same cache dir.
    cache_train = None
    cache_val = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        gs_tag = "gray" if args.grayscale else "rgb"
        state_tag = f"_s{args.state_dim}" if args.state_dim > 0 else ""
        cache_train = cache_dir / f"frames_train_{args.frame_size}_{gs_tag}{state_tag}"
        cache_val = cache_dir / f"frames_val_{args.frame_size}_{gs_tag}{state_tag}"

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

    # pin_memory=True pre-stages batch tensors in page-locked RAM so the
    # GPU DMA transfer is faster. shuffle=True for training, False for val
    # so val metrics are deterministic across runs.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    # Resolve in_channels: default (1) auto-detects from grayscale flag,
    # but an explicit --in-channels override takes priority (e.g., for
    # frame-stacked inputs with multiple channels).
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

    # --- Resume from checkpoint ---
    # Restores model weights, optimizer state (including momentum buffers and
    # LR), epoch counter, and global_step so training continues seamlessly.
    start_epoch = 0
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("global_step", 0)
        print(f"  Restored epoch={start_epoch}, global_step={start_step}")

    # LR scheduler (optional) — ReduceLROnPlateau lowers the learning rate
    # when val loss plateaus, which helps squeeze out final reconstruction
    # quality. Disabled by default (lr_patience=0) because most runs
    # converge fine with a fixed LR.
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min,
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

    # --- Callbacks ---
    # Order matters: validation first (updates best checkpoint + early stop
    # flag), then periodic checkpoints, then diagnostics.
    callbacks = [
        # Runs full val pass every val_every steps; saves best.pt and tracks
        # patience for early stopping.
        PixelVAEValidationCallback(
            val_loader=val_loader, beta=args.beta,
            fg_weight=args.fg_weight,
            state_weight=args.state_weight if args.state_dim > 0 else 0.0,
            every_n_steps=args.val_every, patience=args.patience,
            checkpoint_dir=ckpt_dir,
        ),
        # Periodic checkpoint (independent of best — for crash recovery)
        CheckpointCallback(checkpoint_dir=ckpt_dir, every_n_steps=args.ckpt_every),
        # Logs a grid of input|reconstruction pairs to TensorBoard for
        # visual inspection of VAE quality during training.
        ReconGridCallback(val_loader=val_loader, every_n_steps=args.val_every),
        # Logs gradient norm to detect exploding/vanishing gradients
        GradNormCallback(every_n_steps=50),
        # Halts training immediately if loss goes NaN — faster feedback
        # than waiting for the epoch to finish with garbage gradients.
        NaNDetectionCallback(),
        ProgressCallback(every_n_steps=100, total_epochs=args.epochs),
    ]

    ctx = CallbackContext(
        model=vae, optimizer=optimizer, writer=writer,
        global_step=start_step, epoch=start_epoch, run_dir=str(vae_dir),
        device=args.device,
        extras={"config": config, "scheduler": scheduler},
    )

    # --- SIGINT handler ---
    # Catch Ctrl-C to save an emergency checkpoint before exiting, so long
    # training runs aren't completely lost. Second Ctrl-C force-exits
    # immediately in case the save itself hangs.
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

    remaining = args.epochs - start_epoch
    print(f"\nTraining VAE for {remaining} remaining epochs "
          f"(start_epoch={start_epoch}, total={args.epochs}) on {args.device}")
    for epoch in range(start_epoch, args.epochs):
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

        # Log current LR so we can see scheduler reductions in TensorBoard
        if ctx.writer:
            ctx.writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], ctx.global_step)

        # NOTE: LR scheduler now steps inside the training loop via
        # ctx.extras["scheduler"], not here. See pixel_vae_train_epoch.
        # Keeping this comment for history — the epoch-level step was
        # wrong because val_loss updates ~70 times per epoch but the
        # scheduler only saw the last value.

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
