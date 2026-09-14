JAX implementation of CAST / CAST-ρ, a spanning-tree replacement for the Schur clique in approximate Cholesky (https://arxiv.org/abs/2609.09255).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no SuiteSparse / 201-matrix bakeoff, no Gremban expansion, no max-flow IPM, elimination ordering is minimum-degree not AMD.
