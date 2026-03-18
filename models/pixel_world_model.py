# models/pixel_world_model.py
"""Combined VAE + dynamics pixel world model.

Top-level interface for encoding, predicting, and dreaming.
Supports both single-frame (in_channels=1) and stacked-frame
(in_channels=4) modes. For stacked frames, dreaming maintains
a frame buffer to construct stacked inputs from decoded predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.pixel_vae import PixelVAE


class PixelWorldModel(nn.Module):
    """Combined pixel world model: VAE encoder-decoder + latent dynamics.

    Accepts any dynamics module that implements forward(z, action, state)
    and rollout(z_start, actions) — both LatentDynamicsModel (GRU) and
    LatentRSSM satisfy this interface.
    """

    def __init__(self, vae: PixelVAE, dynamics: nn.Module):
        super().__init__()
        self.vae = vae
        self.dynamics = dynamics

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent z (deterministic — returns mu)."""
        was_training = self.vae.training
        self.vae.eval()
        with torch.no_grad():
            z = self.vae.encode(frames)
        self.vae.train(was_training)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z to frames."""
        return self.vae.decode(z)

    def predict_next(self, frames: torch.Tensor, action: torch.Tensor,
                     hidden: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced single-step: encode real frame, predict next, decode."""
        z = self.vae.encode(frames)
        z_next, hidden = self.dynamics(z, action, hidden)
        pred_frame = self.vae.decode(z_next)
        return pred_frame, z_next, hidden

    @torch.no_grad()
    def dream(self, seed_frames: torch.Tensor, actions: torch.Tensor
              ) -> torch.Tensor:
        """Autoregressive dream: encode seed, roll out feeding own predictions.

        For in_channels=1: operates purely in latent space.
        For in_channels>1: maintains frame buffer, re-encodes stacked frames.

        Returns:
            frames: (B, T+1, C, H, W) — seed frame + T predicted frames
        """
        was_training = self.training
        self.eval()

        B, C, H, W = seed_frames.shape
        T = actions.size(1)

        if C == 1:
            z = self.vae.encode(seed_frames)
            z_seq, _ = self.dynamics.rollout(z, actions)
            all_z = z_seq.reshape(B * (T + 1), -1)
            all_frames = self.vae.decode(all_z)
            frames = all_frames.reshape(B, T + 1, 1, H, W)
        else:
            frame_stack = C
            buffer = list(seed_frames.split(1, dim=1))

            z = self.vae.encode(seed_frames)
            hidden = None
            all_frames = [seed_frames]

            for t in range(T):
                z_next, hidden = self.dynamics(z, actions[:, t], hidden)
                pred_single = self.vae.decode(z_next)[:, :1]
                buffer = buffer[1:] + [pred_single]
                stacked = torch.cat(buffer, dim=1)
                all_frames.append(stacked)
                z = self.vae.encode(stacked)

            frames = torch.stack(all_frames, dim=1)

        self.train(was_training)
        return frames

    @torch.no_grad()
    def dream_from_latent(self, z_seed: torch.Tensor, actions: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dream from a latent seed (pure latent-space rollout)."""
        was_training = self.training
        self.eval()

        z_seq, _ = self.dynamics.rollout(z_seed, actions)
        B, Tp1, D = z_seq.shape
        all_z = z_seq.reshape(B * Tp1, D)
        all_frames = self.vae.decode(all_z)
        C, H, W = all_frames.shape[1:]
        frames = all_frames.reshape(B, Tp1, C, H, W)

        self.train(was_training)
        return frames, z_seq
