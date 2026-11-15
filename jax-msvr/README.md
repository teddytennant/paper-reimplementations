# jax-msvr

JAX kernel of Multi-block-Single-probe Variance Reduction (MSVR) for finite-sum coupled compositional optimization (arXiv:2609.15723).

## Install

```
pip install -e ".[dev]"
```

## Tests

```
pytest -q
```

## What this is

A small CPU JAX implementation of the MSVR estimator (eq. 4), the single-point variant (eq. 5), SOX (eq. 3), and MSVRM-v1/v2/v3 (eqs. 6-9) on a synthetic linear-inner quadratic-outer FCCO problem with known g_i oracles.

## What does not match the paper

This is not the ResNet-18 multi-task deep AUC experiment on CIFAR. Stage-wise MSVRM (Algorithm 3), Adam-style rates, and the sample-complexity theorems are not implemented. Inner oracles are synthetic g_i, not deep networks.
