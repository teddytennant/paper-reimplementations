"""A small x-prediction network for the toy flow matching demo.

The paper trains JiT transformers on ImageNet (Table 5). Nothing here is meant
to stand in for that. What carries over is the parameterization: the network
predicts x_1 directly rather than the velocity, because v- and epsilon-
prediction fail in pixel space, and the velocity is recovered as
v_theta = (x_theta - x_t) / (1 - t) (Section 3.1, "Prediction Parameterization").
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def time_features(t):
    """Seven features from a scalar t: t itself plus sin/cos at 1, 2 and 4 Hz."""
    freqs = jnp.asarray([1.0, 2.0, 4.0], dtype=t.dtype)
    ang = 2.0 * jnp.pi * freqs * t
    return jnp.concatenate([t[None], jnp.sin(ang), jnp.cos(ang)])


class Predictor(eqx.Module):
    """MLP mapping (x_t, t) to a prediction of x_1, in NHWC."""

    net: eqx.nn.MLP
    size: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(self, key, size: int = 16, channels: int = 1, width: int = 256, depth: int = 3):
        dim = size * size * channels
        self.net = eqx.nn.MLP(
            in_size=dim + 7,
            out_size=dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.gelu,
            key=key,
        )
        self.size = size
        self.channels = channels

    def __call__(self, x_t, t):
        """One sample: x_t is (H, W, C) and t is a scalar array."""
        flat = jnp.concatenate([x_t.reshape(-1), time_features(t)])
        return self.net(flat).reshape(self.size, self.size, self.channels)


def predict_batch(model, x_t, t):
    """Vmapped call over the batch axis."""
    return jax.vmap(model)(x_t, t)


def velocity(x_pred, x_t, t, clip: float = 0.05):
    """v_theta = (x_theta - x_t) / (1 - t), Section 3.1."""
    denom = jnp.maximum(1.0 - t, clip).reshape((-1, 1, 1, 1))
    return (x_pred - x_t) / denom
