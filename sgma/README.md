Structure-guided quadtree tokenizer and damped accumulation for masked autoencoding, from https://arxiv.org/abs/2609.30682.

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py
```

No gigapixel images, no SAM or UNETR fine-tune, no Dice numbers. The encoder is one transformer block, not the paper's ViT. Masking defaults to sampling reconstruction targets proportional to the damped canvas, which is what the abstract describes; the paper's equation (5) text also says a uniform 75% mask, and `uniform_mask` is included. Images in the demo are synthetic.
