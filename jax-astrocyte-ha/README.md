# jax-astrocyte-ha

JAX implementation of a two-timescale neural-astrocyte architecture that implements a hybrid automaton for evidence accumulation (arXiv:2609.16217).

## Install

```bash
pip install -e ".[tests]"
```

Requires `jax` and `numpy`. Tests need `pytest`.

## Run

```bash
pytest -q
python scripts/smoke.py
```

CPU is enough. Use `JAX_PLATFORMS=cpu`.

## What does not match the paper

Tiny networks (Nx=16, Nh=8, M=2 by default) instead of Nx=100, Nh=20, M=4. Short trials and a brief training loop, not a full optimization. Simplified delayed match-to-sample: each latent context rotates the stimulus angle by 2π c/M rather than the paper's four nonlinear stimulus-target maps. Not 100 random seeds. No bifurcation, attractor-shallowness, or stickiness analysis. Default reward is deterministic reach success; the paper's stochastic bandit with p* is available but unused in the smoke path. The A-RNN critic is a linear value head (architecture not specified in the paper). H_px may be initialized larger than N(0, 0.01^2) so that Eq. 6 modulation is visible at tiny size.
