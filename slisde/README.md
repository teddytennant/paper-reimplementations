JAX kernel of SLiSDE structured linear SDE layers with gated in-flow stacking (arXiv:2609.34831).

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

This is a small CPU kernel, not the paper's functional-calibration benchmark suite. There is no causal convolution on the gate input, no time-feature decoder for the affine coefficients, no block-diagonal specialised scan, and no CEM / KL-wall dual optimiser for the Girsanov controller. Widths, depth and horizon are tiny; reported table numbers are not reproduced.
