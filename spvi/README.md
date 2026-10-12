implementation in JAX of scan-parallel variational inference for latent jump models (arXiv:2610.12986).

```bash
pip install -e .
python -m pytest
```

The recognition net is a small causal MLP, not the paper's pre-norm transformer; no REFIT or USHCN; synthetic smoke only.
