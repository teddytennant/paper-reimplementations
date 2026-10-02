Sequential block reconstruction quantization for vision transformers (https://arxiv.org/abs/2609.34564).

```
pip install -e .
pytest
```

No ViT, no ImageNet, no Hadamard fusion, no GELU-to-ReLU pass, no W4A8 kernels. CPU tests use synthetic linear blocks. Reconstruction is Adam on the weights and per-channel scale through STE, minimizing the paper's normalized global feature loss plus lambda=2 times the local feature loss.
