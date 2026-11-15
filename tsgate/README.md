# tsgate

Timestep-aware gated attention: a SiLU timestep bias added to content logits before the sigmoid, in JAX on CPU.

```
pip install -e .
```

```
JAX_PLATFORMS=cpu python -m pytest
```

This package does not train the paper's 1.25B diffusion transformer, and it does not report DPG, GenEval, or T2I-CompBench scores. The timestep embedding is an input; the backbone embedding is not reproduced. Audio streams and mixture-of-experts feed-forwards are not included.
