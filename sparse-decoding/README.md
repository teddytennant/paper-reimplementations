# sparse-decoding

SparseDecoding builds layer calibration matrices from autoregressive decode activations (prefill columns dropped), prunes each row with OBS under an N:M mask, and applies the sparse weights with a bitmask SpMV.

```bash
pip install -e .
pytest
```

This package does not load LLM weights, does not download calibration text, and has no Triton kernel. The SpMV is a CPU reference of bitmask indexing and fixed-step least-significant-set-bit traversal. `damp` is a numerical stabilizer added to the Hessian before inversion, not a published constant.
