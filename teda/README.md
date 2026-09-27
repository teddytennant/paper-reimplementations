TEDA from [arXiv:2609.28967](https://arxiv.org/abs/2609.28967) (Chen et al.) is a JAX kernel for passive LWIR hyperspectral ranging: Stage I extracts atmospheric transmittance with a closed-form two-branch absorption-blind baseline, and Stage II estimates range by contrast-weighted alignment to an operator-consistent Beer-Lambert plus ISRF model bank.

```
pip install -e ".[dev]"
pytest -q
python train.py
```

What does not match the paper: HITRAN, ECOSTRESS, and DARPA measured scenes are not used. Absorption is a sum of synthetic Lorentzian lines on 8-13 um, not a line-by-line atmosphere. The supplementary gate extras (quantization, sign guard, dead zone, weighted auxiliary criterion) are omitted; the noise scale is tau_g = c * sigma / median(|Y|) with a small positive floor. CPU only.
