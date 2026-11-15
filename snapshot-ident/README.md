JAX kernel of the weak-form snapshot drift estimator from Beyond Gradient Flow (arXiv:2609.35060).

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

Not d=12 / n=40k. No GLS, no truncated-SVD, no strong-form score fitting, no rate-exponent measurement, and no lineage-tracked particles. Unweighted ridge only. Table 1 numbers are not reproduced.
