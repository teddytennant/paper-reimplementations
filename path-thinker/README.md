implementation in JAX of Painter and Thinker (arXiv 2610.09876): https://arxiv.org/abs/2610.09876

```
pip install -e .
pytest
JAX_PLATFORMS=cpu python scripts/smoke.py
```

The Thinker refines y and z inside each denoising step and adds a ControlNet residual into the Painter skips. This is a tiny 2-level Painter and a 1-block Thinker, not the paper's 512-d 8-head network, and it does not train on the paper's puzzle, maze, or image benchmarks. z resets between denoising steps in the default sampler. An efficient variant carries state and reuses the last control after the midpoint.
