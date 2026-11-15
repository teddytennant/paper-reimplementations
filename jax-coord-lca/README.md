JAX implementation of CoordFormer's coordinate decoder and localized cross-attention (https://arxiv.org/abs/2609.08660).

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tests]"
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python scripts/smoke.py
```

What does not match the paper: no DINOv3/ViT foundation encoder, no MaSS13K/DIS5K/KPIs, no edge-focused inference, toy encoder and synthetic images only.
