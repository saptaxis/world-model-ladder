# World Model Ladder

A minimal, environment-agnostic toolkit for training and evaluating world models — from linear baselines to latent stochastic simulators.

> **Research repo: expect rough edges.** Code here is exploratory and evolves quickly. Expect incomplete experiments and changing APIs.

## Thesis

A *world model* is not a specific architecture. It is a model that learns environment dynamics:

```
s_{t+1} = f(s_t, a_t)
```

Many architectures can serve this role: linear models, MLPs, recurrent networks, latent stochastic models. They differ in inductive bias, not in purpose.

This repository explores a structured ladder of models to answer:

- When does a predictor become a simulator?
- What training objectives enable stable rollouts?
- What data is required for counterfactual correctness?
- Does memory help in fully observed environments?
- When do latent stochastic models become necessary?

## The Ladder

| Level | Model | Training | Key Question |
|-------|-------|----------|-------------|
| 0 | Copy / Linear | Single-step MSE | How much is trivially predictable? |
| 1 | MLP | Single-step MSE | Can a feedforward net learn local transitions? |
| 2 | MLP | Multi-step rollout loss | When does a predictor become a simulator? |
| 3 | GRU | Rollout + scheduled sampling | Does memory help in fully observed environments? |
| 4 | RSSM | ELBO + rollout | When is stochastic latent modeling necessary? |

Each level isolates a specific capability. Models share a common interface, training loop, and evaluation protocol, so comparisons are direct.

## Quick Start

```bash
pip install -r requirements.txt

# Train
python scripts/train.py --config configs/examples/mlp-single-step.yaml

# Evaluate
python scripts/eval.py --checkpoint runs/<run-name>/best.pt
```

Set `data_path` in the config to a directory of `.npz` episode files. Each file should contain `states` (T+1, state_dim) and `actions` (T, action_dim).

The code is fully environment-agnostic — it reads state/action trajectories from `.npz` files and trains transition functions. Any environment that produces trajectories works.

## Model Interface

All models implement `WorldModel.step()`:

```python
delta_pred, next_state = model.step(obs, action, model_state)
#  obs:         [batch, state_dim]   (state-normalized)
#  action:      [batch, action_dim]
#  model_state: None (MLP/Linear) | tensor (GRU) | RSSMState (RSSM)
#  delta_pred:  [batch, state_dim]   (delta-normalized)
```

Models predict **deltas** (change in state), not absolute next states. Normalization is handled externally — models receive normalized inputs and produce normalized outputs.

## Training

Training is step-based and callback-driven. A `CallbackContext` carries shared state; callbacks handle validation, checkpointing, gradient monitoring, rollout evaluation, and more.

```yaml
# configs/examples/mlp-single-step.yaml
arch: mlp
arch_params:
  hidden_dims: [256, 256]
prediction: delta
training_mode: single_step    # or: multi_step, scheduled_sampling, elbo
data_path: SET_ME
```

Training modes:
- **single_step** — MSE on one-step delta prediction
- **multi_step** — MSE on k-step autoregressive rollout (backprop through time)
- **scheduled_sampling** — gradually replace true states with model predictions during rollout
- **elbo** — reconstruction + KL divergence for RSSM

Features: curriculum rollout scheduling, early stopping, step-based checkpoints, NaN detection, graceful Ctrl-C, `--resume`, `--dry_run`.

## Evaluation

All models are evaluated with the same protocol:

- **A. Local prediction** — one-step per-dimension MSE
- **B. Rollout stability** — horizon-error curves and divergence exponent under autoregressive rollout
- **C. Intervention behavior** — response to controlled physics primitives (free fall, sustained thrust, hover)
- **D. Physics variation** — generalization across varying environment parameters (Level 4+)
- **E. Uncertainty calibration** — calibrated predictive distributions (Level 5+)

```bash
python scripts/eval.py --checkpoint runs/<run-name>/best.pt
# Outputs: metrics.json, report.md, horizon/per-dim plots
```

## Pixel World Models

In addition to state-space models, this repo includes a **pixel-based world model pipeline** that learns dynamics directly from video frames.

### Architecture: Staged VAE + Latent Dynamics

Training is split into two phases:

1. **PixelVAE** — convolutional VAE compresses frames to a latent vector z. Optional auxiliary state prediction head forces the latent space to encode physical state (position, velocity, angle).
2. **Latent dynamics** — a recurrent model predicts next latent codes given current z and action. Two dynamics options:
   - **GRU** (`LatentDynamicsModel`) — simple GRU dynamics with `latent_mse` or `multi_step_latent` training
   - **RSSM** (`LatentRSSM`) — stochastic latent-space RSSM with `latent_elbo` (posterior/prior KL) or `multi_step_latent` (prior-only ablation)

### Training Modes

| Model | Training Mode | Description |
|-------|--------------|-------------|
| GRU | `latent_mse` | Teacher-forced predict_sequence + MSE (default) |
| GRU | `multi_step_latent` | k-step autoregressive rollout with full gradient flow |
| RSSM | `multi_step_latent` | Prior-only rollout (skips posterior/KL — ablation) |
| RSSM | `latent_elbo` | Full ELBO: posterior-guided step + KL(posterior ‖ prior) |

### Pixel Training Scripts

```bash
# Phase 1: Train VAE
python scripts/train_pixel_vae.py \
    --data-path /path/to/episodes \
    --run-dir runs/pixel-vae \
    --fg-weight 5.0 --state-dim 6

# Phase 2: Train dynamics (GRU default)
python scripts/train_pixel_dynamics.py \
    --vae-checkpoint runs/pixel-vae/best.pt \
    --data-path /path/to/episodes \
    --run-dir runs/pixel-dyn

# Phase 2: Train dynamics (RSSM with ELBO)
python scripts/train_pixel_dynamics.py \
    --vae-checkpoint runs/pixel-vae/best.pt \
    --data-path /path/to/episodes \
    --run-dir runs/pixel-rssm \
    --model-type rssm --training-mode latent_elbo \
    --rollout-k 10 --kl-weight 1.0

# Generate dream comparison videos
python scripts/dream_compare.py \
    --vae-checkpoint runs/pixel-vae/best.pt \
    --dynamics-checkpoint runs/pixel-dyn/best.pt \
    --data-path /path/to/episodes \
    --output-dir ~/dreams
```

Key flags: `--fg-weight` (foreground pixel upweighting), `--state-dim` (auxiliary state head), `--model-type gru|rssm`, `--training-mode`, `--rollout-k`, `--kl-weight`, `--sampling-start/end` (scheduled sampling for GRU).

## Project Structure

```
models/          Linear, MLP, GRU, RSSM, CopyState + factory
                 PixelVAE, LatentDynamicsModel, LatentRSSM, PixelWorldModel
training/        Loss functions, training loop, callbacks, rollout, scheduling
                 Pixel losses, pixel loop, pixel callbacks
evaluation/      Metrics (per-dim MSE, horizon curves, divergence exponent)
data/            Episode loading, normalization (NormStats)
                 PixelFrameDataset, PixelEpisodeDataset
viz/             DreamGenerator (dream sequences, comparison videos)
scripts/         train.py, eval.py
                 train_pixel_vae.py, train_pixel_dynamics.py, dream_compare.py
utils/           Config, checkpoints, logging, plotting, reporting
configs/         Example YAML configs for each model type
tests/           278 tests covering models, training, evaluation, integration
```

## License

MIT
