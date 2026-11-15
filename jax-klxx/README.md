JAX implementation of KLXX, log-ratio variation for normalizing-flow Boltzmann generators (https://arxiv.org/abs/2609.09473).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no adaptive staging, no alanine-dipeptide / DW4 / LJ13, no L-BFGS quench, no paper tables.
