JAX down-up walk on high-magnetization Hamming slices for the Sherrington–Kirkpatrick model (https://arxiv.org/abs/2609.08873).

Implemented: Hamming-slice helpers (enumerate X_k^d, convert between index sets and ±1 vectors); the SK/GOE Hamiltonian of Model 1 including 1/sqrt(d) coupling scale and field h; Algorithm 1 down-up walk with the Remark 2 heat-bath up-step; a short MCMC runner. Tiny-d brute-force Boltzmann weights for checks.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```

What does not match the paper: no AT-line experiments, no k^{3/2} measurement-complexity tables, polynomial-time mixing is not proven here, tiny d only, skipped annealing (Algorithm 2 / EstimateZ) and the spike-and-slab posterior sampler (Algorithm 3).
