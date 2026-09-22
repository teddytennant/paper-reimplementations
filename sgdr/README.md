# SGDR

CPU JAX reference for Semantic-Geometric Decoupled Routing, the block-sparse attention router in [arXiv 2609.22884](https://arxiv.org/abs/2609.22884).

Router scores follow the paper and do not include `1/sqrt(d)`. That scale is applied only in the optional sparse attention output.

```bash
pip install -e .
PYTHONPATH=src python -m sgdr.demo
PYTHONPATH=src python -m pytest -q
```

This does not match the paper's experimental stack: no Llama or Qwen checkpoints, no RULER or other long-context benchmark, no Triton kernel, and no reported 128K timings or speedups. The wide-sense-stationary check uses a synthetic stationary residual, not real activations. Literal post-RoPE mean pooling equals the exact block score because pairwise products factor, so the seeded win is against the macro-only score with the structural spectrum fixed at 1.
