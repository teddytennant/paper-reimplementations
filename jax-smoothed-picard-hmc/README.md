JAX implementation of Smoothed Picard Hamiltonian Monte Carlo (https://arxiv.org/abs/2609.06906).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```

What does not match the paper: no FORS, no recursive warm start, no proximal BPS, no complexity-theorem parameter tuning, CPU tests on quadratics only.
