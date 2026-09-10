# jax-mars-clip

Implementation in JAX of MARS-CLIP (https://arxiv.org/abs/2609.08283).

```
uv pip install -e ".[dev]"
python -m pytest -q
```

This repository implements the training-free equations on arrays. It does not include a CLIP checkpoint, PASCAL or COCO numbers, PAMR post-processing, or sliding-window evaluation.
