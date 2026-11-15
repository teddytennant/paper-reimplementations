"""Two-layer GraphSAGE encoder with a mean aggregator (no neighbor sampling)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _glorot(key, fan_in, fan_out):
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)


def init_graphsage(key, in_dim, hidden_dim, out_dim):
    """Initialize two GraphSAGE layers. Weights act on concat(self, mean-neighbors)."""
    k1, k2 = jax.random.split(key)
    return {
        "w1": _glorot(k1, 2 * in_dim, hidden_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": _glorot(k2, 2 * hidden_dim, out_dim),
        "b2": jnp.zeros((out_dim,)),
    }


def to_dense_adj(adj, n):
    """Undirected dense adjacency. Accepts square (N, N) or COO (2, E)."""
    adj = jnp.asarray(adj)
    if adj.ndim != 2:
        raise ValueError(f"adjacency must be 2-D, got shape {adj.shape}")
    if adj.shape[0] == n and adj.shape[1] == n:
        a = adj.astype(jnp.float32)
    elif adj.shape[0] == 2:
        rows = adj[0].astype(jnp.int32)
        cols = adj[1].astype(jnp.int32)
        a = jnp.zeros((n, n), dtype=jnp.float32)
        a = a.at[rows, cols].add(1.0)
        a = a.at[cols, rows].add(1.0)
        a = jnp.clip(a, 0.0, 1.0)
    else:
        raise ValueError(
            f"adjacency must be dense (N, N) or COO (2, E); got {adj.shape} with N={n}"
        )
    return jnp.maximum(a, a.T)


def _mean_neighbors(x, adj):
    """Row-normalized mean of neighbors. Self-loops are removed before aggregating."""
    adj = adj - jnp.diag(jnp.diag(adj))
    deg = jnp.maximum(adj.sum(axis=-1, keepdims=True), 1.0)
    return (adj @ x) / deg


def _sage_layer(x, adj, weight, bias, relu):
    neigh = _mean_neighbors(x, adj)
    h = jnp.concatenate([x, neigh], axis=-1) @ weight + bias
    if relu:
        h = jax.nn.relu(h)
    return h


def encode(params, x, adj):
    """Inductive GraphSAGE apply. `x` is (N, F); `adj` is dense (N, N) or COO (2, E)."""
    x = jnp.asarray(x)
    n = x.shape[0]
    adj = to_dense_adj(adj, n)
    h = _sage_layer(x, adj, params["w1"], params["b1"], relu=True)
    h = _sage_layer(h, adj, params["w2"], params["b2"], relu=False)
    return h
