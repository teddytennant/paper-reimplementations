# jax-wrp

JAX implementation of Weight-Redundancy Pruning (https://arxiv.org/abs/2609.09883).

## install

```
uv pip install -e ".[dev]"
```

## pytest

```
JAX_PLATFORMS=cpu python -m pytest -q
```

This library runs the WRP equations on synthetic layer weight dicts. It does not load pretrained checkpoints and does not reproduce the nine-task table. Clustering uses a fixed-seed k-means on the Laplacian embedding.
