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

## Project Structure

```
models/          Linear, MLP, GRU, RSSM, CopyState + factory
training/        Loss functions, training loop, callbacks, rollout, scheduling
evaluation/      Metrics (per-dim MSE, horizon curves, divergence exponent)
data/            Episode loading, normalization (NormStats)
scripts/         train.py, eval.py
utils/           Config, checkpoints, logging, plotting, reporting
configs/         Example YAML configs for each model type
tests/           155 tests covering models, training, evaluation, integration
```

## License

MIT
