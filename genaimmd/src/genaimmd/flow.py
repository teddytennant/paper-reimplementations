"""Conditioned RealNVP split-coupling affine flow (3 blocks)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

N_BLOCKS = 3
HIDDEN = (100, 100, 100)
S_CLIP = 5.0


def _init_mlp(key, sizes):
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros((n_out,))
        params.append((w, b))
    return params


def _mlp(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def _masks(dim: int, n_blocks: int):
    masks = []
    m = jnp.arange(dim) % 2
    # identity on even coords in block 0
    id_mask = (m == 0).astype(jnp.float32)
    for i in range(n_blocks):
        masks.append(id_mask if i % 2 == 0 else 1.0 - id_mask)
    return masks


def init_realnvp(
    key,
    n_blocks: int = N_BLOCKS,
    hidden=HIDDEN,
    dim: int = 2,
    cond_dim: int = 1,
):
    """One coupling MLP per block. Input: identity coords + condition c."""
    blocks = []
    in_dim = 1 + cond_dim  # 2D split is always 1 identity coord
    sizes = (in_dim,) + tuple(hidden) + (2,)  # s, t for the 1D transformed coord
    for _ in range(n_blocks):
        key, k = jax.random.split(key)
        blocks.append(_init_mlp(k, sizes))
    return {"blocks": blocks}


def _as_batch(x, c):
    x = jnp.asarray(x)
    squeeze = x.ndim == 1
    x = jnp.atleast_2d(x)
    n = x.shape[0]
    c = jnp.asarray(c, dtype=x.dtype)
    c = jnp.reshape(c, (-1,))
    if c.size == 1:
        c = jnp.full((n,), c.reshape(()))
    c = c.reshape(n, 1)
    return x, c, squeeze


def _coupling(params, x, c, mask, reverse: bool):
    """Affine coupling on (1-mask) channels, conditioned on identity + c."""
    x_id = jnp.sum(x * mask, axis=-1, keepdims=True)
    inp = jnp.concatenate([x_id, c], axis=-1)
    st = _mlp(params, inp)
    s = jnp.clip(st[..., 0:1], -S_CLIP, S_CLIP)
    t = st[..., 1:2]
    s_vec = s * (1.0 - mask)
    t_vec = t * (1.0 - mask)
    if reverse:
        y = (x - t_vec) * jnp.exp(-s_vec)
        logdet = -jnp.sum(s_vec, axis=-1)
    else:
        y = x * jnp.exp(s_vec) + t_vec
        logdet = jnp.sum(s_vec, axis=-1)
    return y, logdet


def forward(params, z, c):
    """x = F_{zx}(z|c), log|det J_zx|."""
    x, c, squeeze = _as_batch(z, c)
    logdet = jnp.zeros(x.shape[0], dtype=x.dtype)
    masks = _masks(x.shape[-1], len(params["blocks"]))
    for block, mask in zip(params["blocks"], masks):
        x, ld = _coupling(block, x, c, mask, reverse=False)
        logdet = logdet + ld
    if squeeze:
        return x[0], logdet[0]
    return x, logdet


def inverse(params, x, c):
    """z = F_{xz}(x|c), log|det J_xz|."""
    z, c, squeeze = _as_batch(x, c)
    logdet = jnp.zeros(z.shape[0], dtype=z.dtype)
    masks = _masks(z.shape[-1], len(params["blocks"]))
    for block, mask in zip(reversed(params["blocks"]), reversed(masks)):
        z, ld = _coupling(block, z, c, mask, reverse=True)
        logdet = logdet + ld
    if squeeze:
        return z[0], logdet[0]
    return z, logdet
