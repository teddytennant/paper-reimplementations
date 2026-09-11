JAX implementation of AUC maximization from biased PU data with confidence (https://arxiv.org/abs/2609.10928).

This package implements the confidence-weighted PU-AUC risk (eqs. 9, 15–17) and Proposition 4.1 (Bayes-optimal ranking under a strictly increasing transform of the posterior). It does not reproduce the eight real-world datasets or the paper's tables.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke_train.py
```
