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

        # Input projection: fuse latent state and action into a single
        # hidden_size vector before feeding to GRU — this lets the GRU
        # operate in a uniform feature space regardless of latent/action dims
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.ReLU(inplace=True),
        )

        # Single-layer GRU maintains temporal context across steps.
        # One layer suffices because the input projection already does
        # nonlinear feature mixing; adding GRU depth adds latency with
        # minimal accuracy gain at these latent dimensions
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Two-layer output projection: hidden_size -> hidden_size -> latent_dim.
        # The intermediate ReLU lets the network learn nonlinear mappings
        # from GRU hidden state to latent predictions (a single linear layer
        # would limit the model to affine transforms of the hidden state)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, latent_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zero-initialized hidden state. Shape: (1, B, hidden_size)."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    # Backward compatibility alias
    init_hidden = initial_state

    def forward(self, z: torch.Tensor, action: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step prediction."""
        if hidden is None:
            hidden = self.initial_state(z.size(0), z.device)

        # Concatenate current latent and action so the model can learn
        # action-conditioned transitions in latent space
        x = torch.cat([z, action], dim=-1)
        x = self.input_proj(x)
        # GRU expects (B, seq_len, features) — unsqueeze to add seq_len=1
        # for single-step mode while reusing the same GRU module as rollout
        x = x.unsqueeze(1)
        output, hidden = self.gru(x, hidden)
        # Remove the seq_len=1 dimension to return (B, hidden_size)
        output = output.squeeze(1)
        # Project GRU output back to latent space — predicts absolute z_next
        z_next = self.output_proj(output)
        return z_next, hidden

    def rollout(self, z_start: torch.Tensor, actions: torch.Tensor,
                hidden: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step autoregressive rollout. Returns (B, T+1, latent_dim) including z_start."""
        T = actions.size(1)
        if hidden is None:
            hidden = self.initial_state(z_start.size(0), z_start.device)

        # Prepend z_start so the output has T+1 entries — downstream code
        # (multi_step_latent_loss) expects the seed at index 0 for alignment
        # with the ground-truth sequence
        z_seq = [z_start]
        z = z_start
        for t in range(T):
            # Each step feeds its own prediction as input (autoregressive),
            # so errors compound — this matches the dreaming regime
            z, hidden = self.forward(z, actions[:, t], hidden)
            z_seq.append(z)

        # Stack along dim=1 to produce (B, T+1, latent_dim)
        return torch.stack(z_seq, dim=1), hidden

    def predict_sequence(self, z_sequence: torch.Tensor, actions: torch.Tensor,
                         hidden: torch.Tensor | None = None,
                         teacher_forcing: float = 1.0
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next latents for a sequence (training with scheduled sampling)."""
        B, T, _ = z_sequence.shape
        if hidden is None:
            hidden = self.initial_state(B, z_sequence.device)

        z_preds = []
        # Start from the first ground-truth latent as the seed
        z = z_sequence[:, 0]

        for t in range(T):
            z_next, hidden = self.forward(z, actions[:, t], hidden)
            z_preds.append(z_next)

            if t < T - 1:
                # Scheduled sampling: with probability teacher_forcing, use
                # ground-truth z as the next input (stabilises early training);
                # otherwise use the model's own prediction (matches test-time
                # autoregressive regime). Gradually reducing teacher_forcing
                # during training bridges the train/test distribution gap.
                if self.training and torch.rand(1).item() < teacher_forcing:
                    z = z_sequence[:, t + 1]
                else:
                    # Detach during training to prevent gradients from flowing
                    # back through the entire autoregressive chain, which would
                    # cause memory issues and unstable training
                    z = z_next.detach() if self.training else z_next

        return torch.stack(z_preds, dim=1), hidden
