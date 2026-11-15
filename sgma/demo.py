"""Synthetic CPU demo: quadtree tokens, damped canvas, three SGD steps."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sgma.damped import canvas
from sgma.mae import init_params, train_step
from sgma.tokenizer import tokenize_with_depths


def main():
    salience = np.zeros((32, 32), dtype=np.float64)
    salience[:16, :16] = 1.0
    salience[:8, :8] = 4.0

    tokens, depths = tokenize_with_depths(salience, 13, deterministic=True)
    variances = canvas(salience, tokens, depths)
    print(f"token_count {len(tokens)}")
    print(f"mean_canvas_variance {float(variances.mean()):.6f}")

    image = salience.astype(np.float32)
    noise = np.random.default_rng(0).normal(0.0, 0.05, size=image.shape).astype(np.float32)
    images = np.stack([image, np.clip(image + noise, 0.0, None)], axis=0)[..., None]

    import jax

    params = init_params(jax.random.PRNGKey(0), channels=1)
    key = jax.random.PRNGKey(1)
    for step in range(3):
        params, loss = train_step(params, images, salience, jax.random.fold_in(key, step))
        print(f"loss {float(loss):.6f}")


if __name__ == "__main__":
    main()
