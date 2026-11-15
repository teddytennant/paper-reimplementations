Budgeted quotient-residual guidance (QRG) as displayed in arXiv:2609.31222, on toy coordinate quotients.

https://arxiv.org/abs/2609.31222

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[pytest]"
```

## Run

```
.venv/bin/python train.py
.venv/bin/pytest
```

## What does not match the paper

The kernel locks the Euclidean special case of the horizontal lift, sampler-relative budget cap, product-budget split, PredNext evaluation state, and SE(3) equivariance of an invariant distance chart. It does not run pocket-conditioned molecular diffusion, docking, rollout-teacher queries, or the paper's local-radius dense-pair surrogate; Local-QRG here means evaluating the residual at the current state. The combined section-plus-residual step is additionally limited by remaining sampler budget so the delivered pair cannot exceed ρ∥Δx_s∥, whereas equation (14) caps the two branches independently.
