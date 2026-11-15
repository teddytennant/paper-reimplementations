JAX implementation of RiVaT-Fuse variational tensor fusion (https://arxiv.org/abs/2609.10798).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no retinal image-metadata benchmark, no DRO training, no image/metadata CNN encoders, synthetic embeddings only.
