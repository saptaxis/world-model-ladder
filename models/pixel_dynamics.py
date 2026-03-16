# models/pixel_dynamics.py
"""GRU-based dynamics model operating in VAE latent space.

Predicts next latent z_{t+1} from current latent z_t and action a_t.
Maintains GRU hidden state for temporal context. Supports single-step,
multi-step rollout, and teacher-forced sequence prediction with
configurable scheduled sampling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentDynamicsModel(nn.Module):
    """GRU dynamics in VAE latent space.

    Architecture:
        Input projection: Linear(latent_dim + action_dim -> hidden_size) + ReLU
        GRU: hidden_size, 1 layer
        Output projection: Linear(hidden_size -> hidden_size) + ReLU -> Linear(hidden_size -> latent_dim)
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 2,
                 hidden_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, latent_dim),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zero-initialized hidden state. Shape: (1, B, hidden_size)."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, z: torch.Tensor, action: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step prediction."""
        if hidden is None:
            hidden = self.init_hidden(z.size(0), z.device)

        x = torch.cat([z, action], dim=-1)
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        output, hidden = self.gru(x, hidden)
        output = output.squeeze(1)
        z_next = self.output_proj(output)
        return z_next, hidden

    def rollout(self, z_start: torch.Tensor, actions: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step autoregressive rollout. Returns (B, T+1, latent_dim) including z_start."""
        T = actions.size(1)
        if hidden is None:
            hidden = self.init_hidden(z_start.size(0), z_start.device)

        z_seq = [z_start]
        z = z_start
        for t in range(T):
            z, hidden = self.forward(z, actions[:, t], hidden)
            z_seq.append(z)

        return torch.stack(z_seq, dim=1), hidden

    def predict_sequence(self, z_sequence: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor | None = None,
                         teacher_forcing: float = 1.0
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next latents for a sequence (training with scheduled sampling)."""
        B, T, _ = z_sequence.shape
        if hidden is None:
            hidden = self.init_hidden(B, z_sequence.device)

        z_preds = []
        z = z_sequence[:, 0]

        for t in range(T):
            z_next, hidden = self.forward(z, actions[:, t], hidden)
            z_preds.append(z_next)

            if t < T - 1:
                if self.training and torch.rand(1).item() < teacher_forcing:
                    z = z_sequence[:, t + 1]
                else:
                    z = z_next.detach() if self.training else z_next

        return torch.stack(z_preds, dim=1), hidden
