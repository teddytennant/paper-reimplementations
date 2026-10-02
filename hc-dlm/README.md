Hierarchical continuous diffusion language model (arXiv 2610.02193), a tiny JAX slice: the variational token-and-latent loss, the x0-prediction flow step, and an absorbing-mask reverse step.

```
/home/nixos/arxiv-impl-work/bin/py -m pip install -e .
/home/nixos/arxiv-impl-work/bin/py -m pytest -q
```

Does not match the paper: no Sudoku, no Countdown, no LM1B, no transformer stack. The discrete kernel is an absorbing mask kernel. The denoiser is a one-hidden-layer MLP.
