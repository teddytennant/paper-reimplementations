implementation in JAX of FlashBoB (arXiv:2609.24089)

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m flashbob.demo
```

What does not match the paper: no CUDA or Triton kernel, no FlashAttention dependency, no N=262144 run, no 6.3x speedup. Br and Bc are caller arguments, not an SRAM capacity. The traffic counter is the Appendix H element ledger, not a measurement of XLA traffic.
