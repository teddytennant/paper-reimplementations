"""Parametric Sinkhorn matcher: embeddings -> cost -> doubly-stochastic T."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gwsgp.sinkhorn import sinkhorn

PE_DIM = 4


def _linear(key, in_dim: int, out_dim: int):
    k_w, k_b = jax.random.split(key)
    w = jax.random.normal(k_w, (in_dim, out_dim)) / jnp.sqrt(float(in_dim))
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def _dense(params, x):
    return x @ params["w"] + params["b"]


def laplacian_pe(adj: jax.Array, dim: int = PE_DIM) -> jax.Array:
    """Smallest non-trivial Laplacian eigenvectors."""
    deg = jnp.sum(adj, axis=-1)
    lap = jnp.diag(deg) - adj
    _evals, evecs = jnp.linalg.eigh(lap)
    pe = evecs[..., 1 : 1 + dim]
    missing = dim - pe.shape[-1]
    if missing > 0:
        pe = jnp.pad(pe, ((0, 0), (0, missing)))
    return pe


def init_matcher(
    key: jax.Array,
    n_nodes: int,
    n_classes: int,
    embed_dim: int = 16,
    hidden: int = 32,
) -> dict:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    pred_in = n_classes + n_nodes + n_nodes
    tgt_in = n_nodes + PE_DIM + 1
    return {
        "phi1": _linear(k1, pred_in, hidden),
        "phi2": _linear(k2, hidden, embed_dim),
        "psi1": _linear(k3, tgt_in, hidden),
        "psi2": _linear(k4, hidden, embed_dim),
    }


def _node_embeddings_pred(params, x, adj_hat):
    """Position-specific embeddings of the predicted graph.

    Features for node i: class vector, one-hot index, predicted adjacency row.
    """
    n = adj_hat.shape[-1]
    eye = jnp.eye(n, dtype=adj_hat.dtype)
    x_b = jnp.broadcast_to(x, (n, x.shape[-1]))
    feats = jnp.concatenate([x_b, eye, adj_hat], axis=-1)
    h = jax.nn.relu(_dense(params["phi1"], feats))
    return _dense(params["phi2"], h)


def _node_embeddings_tgt(params, adj_star):
    """Structure embeddings of the target graph (row + degree + Laplacian PE)."""
    pe = laplacian_pe(adj_star, dim=PE_DIM)
    deg = jnp.sum(adj_star, axis=-1, keepdims=True)
    feats = jnp.concatenate([adj_star, pe, deg], axis=-1)
    h = jax.nn.relu(_dense(params["psi1"], feats))
    return _dense(params["psi2"], h)


def matching_cost(params, x, adj_hat, adj_star) -> jax.Array:
    """Squared Euclidean cost ``C_ij = ||phi(u_i) - psi(v_j)||^2``."""
    phi = _node_embeddings_pred(params, x, adj_hat)
    psi = _node_embeddings_tgt(params, adj_star)
    diff = phi[:, None, :] - psi[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def match(
    params,
    x,
    adj_hat,
    adj_star,
    tau: float = 0.1,
    n_iters: int = 24,
) -> jax.Array:
    """Amortized transport plan via Sinkhorn on embedding distances."""
    cost = matching_cost(params, x, adj_hat, adj_star)
    return sinkhorn(cost, epsilon=tau, n_iters=n_iters)


def omega_bistochastic(plan: jax.Array) -> jax.Array:
    """Penalty for deviation from the Birkhoff polytope (row/column sums)."""
    row = jnp.sum(plan, axis=-1)
    col = jnp.sum(plan, axis=-2)
    ones = jnp.ones_like(row)
    return jnp.sum((row - ones) ** 2) + jnp.sum((col - ones) ** 2)
