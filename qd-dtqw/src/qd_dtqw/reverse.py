"""Tiny classical reverse denoiser (paper §II.C, without a full U-Net).

A 2-layer SAME conv on one-hot intensities, with a sinusoidal time embedding
summed into the hidden features. Output logits over K classes per pixel.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from qd_dtqw.losses import categorical_cross_entropy, log_softmax, softmax


class ReverseParams(NamedTuple):
    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray
    tw: jnp.ndarray
    tb: jnp.ndarray

    @property
    def k(self) -> int:
        return int(self.w2.shape[-1])

    @property
    def hidden(self) -> int:
        return int(self.w1.shape[-1])


def _sinusoidal_time(t, dim: int, max_period: float = 64.0):
    t = jnp.asarray(t, dtype=jnp.float64)
    half = dim // 2
    freqs = jnp.exp(
        -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float64) / jnp.maximum(half - 1, 1)
    )
    ang = t.reshape(-1, 1) * freqs.reshape(1, -1)
    emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def _conv_same(x, w, b):
    return (
        jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + b
    )


def init_reverse(key, k: int, hidden: int = 8, kernel: int = 3) -> ReverseParams:
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.1
    w1 = scale * jax.random.normal(k1, (kernel, kernel, k, hidden), dtype=jnp.float64)
    b1 = jnp.zeros((hidden,), dtype=jnp.float64)
    w2 = scale * jax.random.normal(k2, (kernel, kernel, hidden, k), dtype=jnp.float64)
    b2 = jnp.zeros((k,), dtype=jnp.float64)
    tw = scale * jax.random.normal(k3, (hidden, hidden), dtype=jnp.float64)
    tb = jnp.zeros((hidden,), dtype=jnp.float64)
    return ReverseParams(w1=w1, b1=b1, w2=w2, b2=b2, tw=tw, tb=tb)


def reverse_logits(params: ReverseParams, x, t):
    """Logits of p(x_t | x_{t+1}, t). x is integer (..., H, W) in {0,...,K-1}."""
    x = jnp.asarray(x)
    squeezed = False
    if x.ndim == 2:
        x = x[None, ...]
        squeezed = True
    if x.ndim != 3:
        raise ValueError("x must have shape (H, W) or (B, H, W)")
    k = params.k
    h = jax.nn.one_hot(x.astype(jnp.int32), k, dtype=jnp.float64)
    h = _conv_same(h, params.w1, params.b1)
    t_emb = _sinusoidal_time(t, params.hidden)
    if t_emb.shape[0] == 1 and h.shape[0] > 1:
        t_emb = jnp.broadcast_to(t_emb, (h.shape[0], params.hidden))
    t_feat = t_emb @ params.tw + params.tb
    h = jax.nn.relu(h + t_feat[:, None, None, :])
    logits = _conv_same(h, params.w2, params.b2)
    if squeezed:
        logits = logits[0]
    return logits


def reverse_step(params: ReverseParams, x, t, key=None, sample: bool = True):
    """One reverse transition. Output stays in {0,...,K-1} by construction."""
    logits = reverse_logits(params, x, t)
    if sample:
        if key is None:
            raise ValueError("key is required when sample=True")
        return jax.random.categorical(key, logits)
    return jnp.argmax(logits, axis=-1)


def generate(
    params: ReverseParams,
    shape,
    T: int,
    key,
    sample: bool = True,
):
    """x̂_T ~ Uniform{0,...,K-1}, then iterate t = T-1 … 0 (paper §II.C)."""
    k = params.k
    key, k0 = jax.random.split(key)
    x = jax.random.randint(k0, shape, 0, k)
    for t in range(int(T) - 1, -1, -1):
        if sample:
            key, kt = jax.random.split(key)
            x = reverse_step(params, x, t, key=kt, sample=True)
        else:
            x = reverse_step(params, x, t, sample=False)
    return x


def sgd_step(params: ReverseParams, x_next, x_t, t, lr: float = 0.05) -> tuple[ReverseParams, jnp.ndarray]:
    """One SGD step on categorical CE for a batch of (x_{t+1}, x_t, t)."""

    def loss_fn(p):
        logits = reverse_logits(p, x_next, t)
        return categorical_cross_entropy(logits, x_t, k=p.k)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    new = jax.tree_util.tree_map(lambda w, g: w - lr * g, params, grads)
    return new, loss


# re-export for callers that want softmax of the reverse model
__all__ = [
    "ReverseParams",
    "generate",
    "init_reverse",
    "log_softmax",
    "reverse_logits",
    "reverse_step",
    "sgd_step",
    "softmax",
]
