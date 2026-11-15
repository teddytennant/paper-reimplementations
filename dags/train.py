"""Train the stabilizer conv stand-in on a synthetic lighting target.

The target is a function of the demodulated frame (a fixed box blur of that
frame), so the conv has something it can fit. Encoder weights are not trained.
Flux and the path-traced dataset are not used. Defaults to CPU unless
JAX_PLATFORMS is already set.
"""

import os
import sys
from pathlib import Path

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp

from dags.control import demodulate
from dags.modules import StabilizerStep, box_blur_3x3


def main():
    key_albedo, key_image, key_params = jax.random.split(jax.random.PRNGKey(0), 3)
    albedo = jax.random.uniform(key_albedo, (8, 8, 3), minval=0.3, maxval=0.8)
    image = jax.random.uniform(key_image, (8, 8, 3), minval=0.6, maxval=1.5)
    frame = demodulate(image, albedo)
    # Blur of the demodulated frame: a fixed function of the input.
    target = box_blur_3x3(frame)
    motion = jnp.zeros((8, 8, 2))
    valid = jnp.ones((8, 8))
    L_prev = jnp.zeros((8, 8, 3))
    params = StabilizerStep.init(key_params, channels=3, std=0.02)

    lr = 0.05
    steps = 30

    def loss_fn(p):
        pred = StabilizerStep.apply(p, frame, motion, valid, L_prev)
        return jnp.mean((pred - target) ** 2)

    @jax.jit
    def update(p):
        grads = jax.grad(loss_fn)(p)
        return jax.tree.map(lambda w, g: w - lr * g, p, grads)

    first = None
    last = None
    for step in range(steps + 1):
        loss_v = float(loss_fn(params))
        if step % 10 == 0:
            print(f"step {step} loss {loss_v:.6f}")
        if step == 0:
            first = loss_v
        last = loss_v
        if step == steps:
            break
        params = update(params)

    if last < first:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
