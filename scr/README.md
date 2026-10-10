# scr

Anchor-based clustering of query sub-chunks so selective attention retrieves once per cluster instead of once per sub-chunk, in JAX on CPU.

```
pip install -e .
```

```
JAX_PLATFORMS=cpu python -m pytest
```

This package does not run vLLM, FlashInfer, or the DeepEdu tutoring agent. Fingerprints are mean-pooled queries, and the selection score is a sum of per-head softmaxes. It does not train a model.
