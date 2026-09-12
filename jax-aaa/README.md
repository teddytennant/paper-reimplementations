JAX implementation of Adaptive Anisotropic Attention (https://arxiv.org/abs/2609.08788).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no EEG MAE pretraining, no TUH-EEG / I-CARE, no 22-layer AXON, toy grid classification only.
