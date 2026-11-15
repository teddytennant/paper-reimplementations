implementation in JAX of SGDR (arXiv:2609.22884)

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m sgdr.demo
```

Router scores do not include `1/sqrt(d)`. That scale is applied only in the optional sparse attention output.

What does not match the paper: no Llama or Qwen checkpoints, no long-context benchmark, no Triton kernel, no reported speedups. The wide-sense-stationary check uses a synthetic residual, not real activations. Literal post-RoPE mean pooling equals the exact block score because pairwise products factor, so the seeded win is against the macro-only score with the structural spectrum fixed at 1.
