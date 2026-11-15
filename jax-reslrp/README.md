JAX kernel of ResLRP residual-aware LRP-γ on skip additions (arXiv:2609.17152).

Install: `pip install -e ".[dev]"`

Run: `JAX_PLATFORMS=cpu python -m pytest -q` and `JAX_PLATFORMS=cpu python scripts/smoke.py`

What does not match the paper: no ViT, AttnLRP, CLIP, ImageNet, FunnyBirds, SAE, or VLM; only a 2-block GELU residual MLP on synthetic tensors. ResLRP+/DAVE is omitted. Eq. 3 has no ε; `lrp_gamma(..., eps)` is a degenerate-denominator guard (equal split when the γ-weighted sum is ~0) so conservation stays exact. `cancellation` is Eq. 2 (C=1 means no cancellation), not a min/max opposite-sign heuristic. `amplification(z_in, z_out)` is the forward mass ratio Σ|z_in|/Σ|z_out|; paper A on relevances is `relevance_amplification`. Linear LRP-ε uses the z-rule and redistributes bias/ε leftover so the residual MLP conserves. GELU uses the identity rule (AttnLRP-style).
