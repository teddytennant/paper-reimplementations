Exact backward-over-backward for single-head softmax attention ([arXiv:2609.24089](https://arxiv.org/abs/2609.24089)): the two-pass tiled schedule of Algorithm 2, plus a dense reference of the same closed forms.

```
pip install -e .
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m flashbob.demo
```

`flashbob_dense` and `flashbob_tiled` take Q, K, V, dO, UQ, UK, UV and return the gradient of Phi = <dQ, UQ> + <dK, UK> + <dV, UV> with respect to Q, K, V, and dO. A boolean mask marks valid scores (True keeps the entry). `causal_mask(n)` keeps column j visible to row i when j <= i. tau is 1/sqrt(d).

What does not match the paper: there is no CUDA or Triton kernel, no FlashAttention dependency, and no attempt to reproduce the N=262144 run or the 6.3x speedup. Block sizes Br and Bc are caller arguments, not chosen from an SRAM capacity M. The traffic counter is the Appendix H element ledger of the schedule, not a measurement of XLA memory traffic. Multi-head attention, dropout, sliding-window kernels, and the training experiments are out of scope.
