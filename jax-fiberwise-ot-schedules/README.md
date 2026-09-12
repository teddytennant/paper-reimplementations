JAX implementation of fiberwise optimal-transport schedules (https://arxiv.org/abs/2609.11842).

```
uv venv .venv --python 3.12
.venv/bin/pip install -e ".[tests]"
export JAX_PLATFORMS=cpu
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke.py
```

What does not match the paper: no CIFAR FID, no U-Net, synthetic risk only, no frozen analytic template unless a few lines.
