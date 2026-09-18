# jax-admissable

JAX implementation of adversarial-missingness training for RL observations (arXiv:2609.15297).

## Install

```
pip install -e ".[tests]"
```

## Run

```
python scripts/smoke.py
pytest -q
```

## Differences from the paper

The paper trains SAC on MuJoCo Hopper, Walker, and Ant. This package does not use MuJoCo. It provides the missingness mask and p_miss update from the paper, a small SAC, and a toy continuous environment with a planted critical observation coordinate (the toy env is not in the paper). Training defaults use far fewer steps than the paper (hundreds rather than millions). Hyperparameters zeta=5000, epsilon=1e-8, l=0 match Table 1; experiments here typically use a smaller zeta so a short run actually updates p_miss.
