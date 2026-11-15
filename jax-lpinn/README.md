JAX implementation of lPINN (arXiv:2609.14926): pretrained neural basis, online last-layer residual least squares.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
```

On NixOS, `import jax` may need `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH`.

## Run

```bash
JAX_PLATFORMS=cpu pytest -q
python scripts/smoke.py
```

## What does not match the paper

- Defaults are tiny (Ntrain=8, Nη=8, Nx=16, Nt=8, a handful of Adam steps) versus the paper's Ntrain=500, Nη=50, Nx=100, Nt=30 and long offline training.
- No pendulum, no Fourier-feature network.
- No Burgers / Cole–Hopf experiments.
- No PINN timing tables or paper error magnitudes.
- Coordinates are affinely scaled to [-1, 1] using L and T before the MLP; the paper does not write this scaling.
- Two tanh hidden layers of width Nη; the paper uses that width in every hidden layer but does not fix the depth in the ADE section.
