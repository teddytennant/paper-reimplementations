JAX implementation of DKRGD (https://arxiv.org/abs/2609.11712).

This implements local robust kernel gradient descent and sample-size-weighted aggregation on synthetic data. It does not reproduce the paper's learning-rate tables or figures.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke_train.py
```

The Newton–Raphson communication scheme in §2.2 (eq. 12) is not implemented: an exact step needs the global operators \(L_{K,D}\), \(\hat f_{K,D}\) and \(\{E_{i,D,\sigma}\}\), and the paper leaves efficient distributed estimation of those quantities as future work.
