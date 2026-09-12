JAX implementation of DataShifts (https://arxiv.org/abs/2609.11918).

Implements the plug-in and debiased entropic OT X-shift estimators, the γ*-concept-shift estimator, and Algorithm 1. It does not reproduce the Novozymes, PACS, or ColoredMNIST experiments or the paper's tables.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```
