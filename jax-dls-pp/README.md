JAX implementation of dynamic latent-space inhomogeneous Poisson processes (https://arxiv.org/abs/2609.08813).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no ICEWS diplomacy data, no Hutchinson-trace BIC, cubic B-splines with a small knot count, synthetic Poisson events only.
