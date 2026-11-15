implementation in JAX of the exact log-symplectic flow (arXiv:2609.24737)

```
pip install -e '.[tests]'
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m ananke.demo
```

The H200 smoke did not complete. Job 738498 was cancelled after Slurm estimated a start of 2026-09-22T10:39:00, which was after this session.

What does not match the paper: no CIFAR-100, no Kvasir, no 0.27M hierarchical model, no reported accuracies. The context generator is a 1x1 conv stand-in. Depthwise 3x3+5x5 stands in for the multi-scale diffusion operator. The smoke net is a stem plus two blocks, not the paper's pyramid.
