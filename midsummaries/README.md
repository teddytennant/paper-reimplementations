# midsummaries

Instance-optimal adaptive location estimation via multiscale mid-summaries (Wang, Gao; arXiv:2609.20749).

https://arxiv.org/abs/2609.20749

## Install

```bash
pip install -e .
python -m pytest -q
```

## Usage

```python
from midsummaries import estimate, estimate_crossfit

mu = estimate(x, k=2)
mu_cf = estimate_crossfit(x, k=2)
```

`estimate` is Algorithm 1: split the sample, estimate quantile spacings on the holdout half, and return a convex combination of mid-summaries on the training half. `estimate_crossfit` runs Algorithm 1 on both split directions and averages.

## What does not match the paper

This package implements the estimator, not the finite-sample oracle-rate proof, and it does not reproduce the Laplace/uniform Monte Carlo tables from §4. The default rank is k=2 as in the numerical studies, not the theoretical k = ceil(720 log(1/δ)). If n is odd, the leftover last point is dropped so both halves have length floor(n/2). Scales whose estimated gap ℓ̂_j is zero or tiny are skipped (weight 0); if every scale is skipped, the mid-summaries are averaged with equal weights.
