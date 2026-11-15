Implementation in JAX of learnable subspace projections (arXiv:2609.40127).

Orthogonal projectors on frozen linear layers. Unconstrained bases are initialized with a whitened SVD, trained by KL to the dense teacher plus an orthogonality penalty, and merged to low-rank factors. Bias is never trained and never projected.

## Install

```bash
pip install -e .
```

## Tests

```bash
pytest
```

## What this is not

No OPT, Qwen, Llama, or ViT models. No perplexity numbers and no speed claims.
