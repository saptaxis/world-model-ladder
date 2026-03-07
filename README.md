# World Model Ladder

A minimal, environment-agnostic research framework for building and evaluating world models, from simple transition predictors to latent stochastic simulators.

> **Research repo: expect rough edges.** Code here is exploratory and evolves quickly. Expect incomplete experiments and changing APIs.

This repository implements a ladder of increasingly expressive dynamics models for learning environment transitions. The goal is to clarify what actually makes a *world model*.

The code is fully environment-agnostic — it reads state/action trajectories from `.npz` files and trains transition functions. Any environment that produces `(states, actions)` sequences works. The first testbed is a custom fork of Gymnasium's LunarLander-v3 that parameterizes 7 physics variables (gravity, thrust power, density, damping, wind), but the models, training, and evaluation code know nothing about the environment.

## Thesis

A *world model* is not a specific architecture.

It is a model that learns environment dynamics:

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

## The World Model Ladder

| Level | Model | Training | Key Question |
|-------|-------|----------|-------------|
| 0 | Copy / Linear | Single-step supervised | How much is trivially predictable? |
| 1 | MLP | Single-step supervised | Can we learn local transitions? |
| 2 | MLP | Multi-step rollout loss | When does a predictor become a simulator? |
| 3 | GRU | Rollout + scheduled sampling | Does memory help or memorize behavior? |
| 4 | MLP/GRU + θ | Rollout, varying physics | Can one model learn a family of physics? |
| 5 | RSSM | ELBO + rollout | When is stochastic latent modeling necessary? |

Each level isolates a specific capability required for world models, from local transition prediction to stochastic latent simulation.

## Evaluation

All models are evaluated with the same protocol for direct comparison:

- **A. Local prediction:** one-step per-dimension MSE
- **B. Rollout stability:** horizon-error curves and divergence exponent under autoregressive rollout
- **C. Intervention behavior:** response to controlled physics primitives (free fall, sustained thrust, hover)
- **D. Physics variation:** generalization across varying environment parameters (Level 4+)
- **E. Uncertainty calibration:** calibrated predictive distributions (Level 5+)

## License

MIT
