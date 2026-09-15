JAX implementation of MoARa (arXiv:2609.15037) structure-preserving block decomposition and module-aware rank allocation for low-rank LLM pre-training.

Install: `pip install -e ".[dev]"`

Run: `JAX_PLATFORMS=cpu python -m pytest -q` and `JAX_PLATFORMS=cpu python scripts/smoke.py`

What does not match the paper: no LLM trainer or 350M–7B runs (synthetic tensors only); decompose `eps` defaults to 1e-6 as in the public API (Appendix B.1 uses 1e-12); GaLore scale 0.25 multiplies the reconstructed direction update (Algorithm 1 does not write the scale); SVD is dense `jax.numpy.linalg.svd`, not randomized; `default_block_size` without `scale` returns `d_head` (Appendix D.2 picks B=32 for 350M even though d_head=64).
