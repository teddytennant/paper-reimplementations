# entropfill

Rényi-guided context pruning with a computable deletion envelope, arXiv [2610.09757](https://arxiv.org/abs/2610.09757).

```
python -m pytest
```

Sink-isolated pooling, null-calibrated chunk scores, the mixture-to-row deletion envelope, and the feasible scan. The paper reports no experiments, and neither does this: there is no transformer, no observer audit, and no RAG stack. The Lipschitz constants of Section 6 are omitted.
