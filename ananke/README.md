Exact log-symplectic circular flow and a CTAN-Res block ([arXiv:2609.24737](https://arxiv.org/abs/2609.24737)).

```
pip install -e .
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m ananke.demo
```

## What does not match the paper
- no CIFAR-100, no Kvasir, no 0.27M hierarchical model, no reported accuracies
- context generator is a 1x1 conv stand-in; the paper does not specify G
- depthwise 3x3+5x5 stands in for the multi-scale diffusion operator
- smoke net is a stem plus two blocks, not the paper's pyramid
- dual-shear rotation is implemented but unused by the block; Branch B is not implemented
