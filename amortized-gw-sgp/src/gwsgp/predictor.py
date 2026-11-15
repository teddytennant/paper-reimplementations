"""MLP graph predictor: class vector -> n x n adjacency in (0, 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _linear(key, in_dim: int, out_dim: int):
    k_w, k_b = jax.random.split(key)
    w = jax.random.normal(k_w, (in_dim, out_dim)) / jnp.sqrt(float(in_dim))
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def _dense(params, x):
    return x @ params["w"] + params["b"]


def init_predictor(
    key: jax.Array,
    n_nodes: int,
    n_classes: int,
    hidden: int = 64,
) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "l1": _linear(k1, n_classes, hidden),
        "l2": _linear(k2, hidden, hidden),
        "l3": _linear(k3, hidden, n_nodes * n_nodes),
    }


def predict_adjacency(params, x) -> jax.Array:
    """Decode a symmetric, zero-diagonal adjacency from input ``x``."""
    logits_flat = _dense(params["l3"], jax.nn.relu(_dense(params["l2"], jax.nn.relu(_dense(params["l1"], x)))))
    n = int(round(params["l3"]["w"].shape[1] ** 0.5))
    logits = logits_flat.reshape(n, n)
    logits = 0.5 * (logits + logits.T)
    adj = jax.nn.sigmoid(logits)
    return adj * (1.0 - jnp.eye(n, dtype=adj.dtype))
