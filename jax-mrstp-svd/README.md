JAX implementation of multi-term randomized semi-tensor-product T-SVD (https://arxiv.org/abs/2609.11168).

This implements the tensor semi-tensor product, MSTP-SVD, and MRSTP-SVD on arrays. It does not reproduce the paper's image/video compression or completion experiments.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```

Truncation rank is a single integer applied to every term, not the paper's full matrix \(R\). DCT and random orthogonal transforms from Section 6 are not implemented.
