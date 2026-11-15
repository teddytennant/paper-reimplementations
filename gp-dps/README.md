JAX kernel of calibrated derivative-process sensitivity (DPS) for Gaussian-process variable selection (arXiv:2609.33549).

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

No real-data tables, no 100-rep FDR, Matérn optional not required, inducing-point comparison omitted, default B small.
