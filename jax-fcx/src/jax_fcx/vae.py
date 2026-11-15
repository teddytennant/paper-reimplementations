"""Conditional VAE q_phi(z | x_M, y), p_theta(x_M' | z, y) (Algorithm 1, §5.2)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


Params = dict[str, Any]


def _init_linear(key: jax.Array, d_in: int, d_out: int, scale: float | None = None) -> dict[str, jax.Array]:
    if scale is None:
        scale = (2.0 / max(d_in, 1)) ** 0.5
    return {"W": jax.random.normal(key, (d_in, d_out)) * scale, "b": jnp.zeros((d_out,))}


def _mlp_sizes(d_in: int, hidden: int, d_out: int, n_linear: int) -> list[int]:
    """n_linear weight matrices: hidden ReLUs then a linear head."""
    n_linear = max(int(n_linear), 2)
    return [d_in] + [hidden] * (n_linear - 1) + [d_out]


def _init_stack(key: jax.Array, sizes: list[int]) -> list[dict[str, jax.Array]]:
    layers = []
    n_edges = len(sizes) - 1
    for i, (d_in, d_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        scale = 0.1 if i == n_edges - 1 else None
        layers.append(_init_linear(k, d_in, d_out, scale=scale))
    return layers


def _apply_stack(layers: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
    h = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        h = h @ layer["W"] + layer["b"]
        if i < last:
            h = jax.nn.relu(h)
    return h


def init_cvae(
    key: jax.Array,
    d_mutable: int,
    z_dim: int,
    hidden: int = 32,
    n_linear: int = 5,
    n_classes: int = 2,
) -> Params:
    """Five linear layers each for encoder and decoder (paper §5.2), no batch-norm.

    Decoder last-layer bias is shifted negative so initial x_M' is small; that
    makes the unary age penalty visible before training.
    """
    k_enc, k_dec = jax.random.split(key)
    enc_in = d_mutable + n_classes
    dec_in = z_dim + n_classes
    encoder = _init_stack(k_enc, _mlp_sizes(enc_in, hidden, 2 * z_dim, n_linear))
    decoder = _init_stack(k_dec, _mlp_sizes(dec_in, hidden, d_mutable, n_linear))
    decoder[-1]["b"] = decoder[-1]["b"] - 2.0
    return {"encoder": encoder, "decoder": decoder}


def _one_hot_y(y: jax.Array, n_classes: int = 2) -> jax.Array:
    return jax.nn.one_hot(y.astype(jnp.int32), n_classes)


def encode(params: Params, x_m: jax.Array, y: jax.Array, n_classes: int = 2) -> tuple[jax.Array, jax.Array]:
    """Return (mu, var) with var = sigmoid(raw) in (0, 1) as in §5.2."""
    h = jnp.concatenate([x_m, _one_hot_y(y, n_classes)], axis=-1)
    head = _apply_stack(params["encoder"], h)
    z_dim = head.shape[-1] // 2
    mu = head[..., :z_dim]
    var = jax.nn.sigmoid(head[..., z_dim:])
    return mu, var


def sample_z(key: jax.Array, mu: jax.Array, var: jax.Array) -> jax.Array:
    eps = jax.random.normal(key, mu.shape)
    return mu + jnp.sqrt(jnp.clip(var, 1e-6, 1.0)) * eps


def decode(params: Params, z: jax.Array, y: jax.Array, n_classes: int = 2) -> jax.Array:
    """Decode mutable features into [0, 1]."""
    h = jnp.concatenate([z, _one_hot_y(y, n_classes)], axis=-1)
    return jax.nn.sigmoid(_apply_stack(params["decoder"], h))


def assemble_counterfactual(x: jax.Array, x_m_cf: jax.Array, mutable_idx: jax.Array) -> jax.Array:
    """x' = x_M' union X_I (Algorithm 1 line 9): copy immutables from x."""
    return x.at[:, mutable_idx].set(x_m_cf)


def generate_cf(
    params: Params,
    x: jax.Array,
    y_target: jax.Array,
    mutable_idx: jax.Array,
    key: jax.Array | None = None,
    n_classes: int = 2,
) -> jax.Array:
    """Encode-decode one batch. Uses mu when key is None."""
    x_m = x[:, mutable_idx]
    mu, var = encode(params, x_m, y_target, n_classes=n_classes)
    z = mu if key is None else sample_z(key, mu, var)
    x_m_cf = decode(params, z, y_target, n_classes=n_classes)
    return assemble_counterfactual(x, x_m_cf, mutable_idx)
