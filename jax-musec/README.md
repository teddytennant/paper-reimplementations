# jax-musec

JAX implementation of Musec / Soft Musec (https://arxiv.org/abs/2609.11655).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke_train.py
```

This repository implements spectral clipping (Algorithm 2) and Soft Musec via coupled Newton–Schulz (Algorithms 3–4) on arrays. It does not include LLM training, FineWeb/OpenWebText/C4 runs, or paper tables.
