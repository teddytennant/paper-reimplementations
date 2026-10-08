# Activation-aware weight tensorization

Training-free diagonal calibration before a tensor-train (TT) decomposition of a linear map, from arXiv 2610.10085 (Beatini, Calebi, Tilli, Benini, Activation-aware Weight Tensorization).

A calibration matrix C with shape (T, n) gives one importance score per input channel. Those scores become positive column scales s. The weight W with shape (m, n) is preconditioned as W' = W diag(s), then factored by TT-SVD. Deployment applies the inverse scales to the input: y = W_hat'(D^{-1} x), computed as a TT matvec of the cores against x / s. Only synthetic linear layers are included.

Run the tests from this directory with the shared venv. The package is not installed into that venv. pytest adds `src` through pythonpath.

```bash
export LD_LIBRARY_PATH="/nix/store/larys5yihddj2diyab60hpdri8n11kn7-ld-library-path/share/nix-ld/lib:/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/home/nixos/arxiv-impl-work/.venv/bin/python -m pytest -q
```

This core does not load Llama, Ministral, or Qwen, and it does not download weights. TTN is not implemented. There is no perplexity evaluation and there are no learned scales. Scales are a closed form of a chosen activation statistic.

The paper ablates alpha in {0.5, 1.0, 1.5} and does not publish a single default for eps or min_scale. The defaults alpha=1.0, eps=1e-6, and min_scale=1e-3 are conveniences for synthetic checks, not paper defaults.

Mode order follows eq 1. W is reshaped in C order to (m1, ..., mK, n1, ..., nK), then transposed to the interleaved order (m1, n1, m2, n2, ..., mK, nK) that the cores contract. The paper's prose also writes the tensor as output modes then input modes. This code follows eq 1.

Std importance is sqrt(Var_t(C[:, i]) + eps). Var_t is the population variance, with ddof=0 (divide by T, not T-1). MeanAbs is the mean of absolute activations and does not use eps.
