# jax-me-decoding

JAX kernel of Mahalanobis-Ensemble Decoding (ME-Decoding) from arXiv:2609.18723.

## Install

```bash
pip install -e ".[tests]"
```

Requires `jax`, `jaxlib`, `numpy`. Tests need `pytest`.

## Tests

```bash
pytest -q
```

CPU is enough.

## What this is

At each decoding step the kernel takes logits `z` and token embeddings `E`, forms `p = softmax(z / T)`, restricts to a top-N candidate pool, builds an RBF kernel on Euclidean distances with median pairwise bandwidth, and greedily grows a subset `S` that maximises the Mahalanobis energy `F(S) = p_S^T K_{S,S}^{-1} p_S` with a rank-1 Cholesky update. Early stopping uses the relative energy gain against `λ (1 - A_S)`. Truncation zeros out logits outside `S` (as `-inf`); sampling is from the renormalised support.

## What does not match the paper

- No LLM, no GSM8K/GPQA numbers.
- The candidate pool is top-N (default 512), not the full vocabulary.
- No HuggingFace LogitsProcessor.
- The authors' PyTorch repository uses a cosine kernel on L2-normalised embeddings and sets the bandwidth to a probability-weighted dispersion divided by 2.5. This package follows the paper RBF on Euclidean distances with median pairwise bandwidth, `K_ij = exp(-||e_i - e_j||^2 / (2 ε_t^2))`, `ε_t = median_{i<j} ||e_i - e_j||_2`.
