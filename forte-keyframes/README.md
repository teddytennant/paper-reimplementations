Training-free keyframe selector in JAX: GP relevance scoring plus exact logarithmic-coverage optimization (arXiv:2610.00573).

```
python3 -m pytest
```

What does not match the paper: no MLLM, no video decoder, no benchmark numbers. The relevance scorer is an injected function. Batched scoring follows the paper's select-score-update rule but does not run a vision model.
