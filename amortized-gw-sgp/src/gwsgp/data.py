"""Synthetic supervised graph prediction data: path, cycle, complete bipartite."""

from __future__ import annotations

import jax
import jax.numpy as jnp


N_CLASSES = 3
CLASS_PATH = 0
CLASS_CYCLE = 1
CLASS_BIPARTITE = 2


def path_adjacency(n: int) -> jax.Array:
    idx = jnp.arange(n - 1)
    adj = jnp.zeros((n, n))
    adj = adj.at[idx, idx + 1].set(1.0)
    adj = adj.at[idx + 1, idx].set(1.0)
    return adj


def cycle_adjacency(n: int) -> jax.Array:
    adj = path_adjacency(n)
    return adj.at[0, n - 1].set(1.0).at[n - 1, 0].set(1.0)


def complete_bipartite_adjacency(n: int) -> jax.Array:
    k = n // 2
    left = jnp.arange(n) < k
    mask = left[:, None] != left[None, :]
    return mask.astype(jnp.float32)


def complete_adjacency(n: int) -> jax.Array:
    return (1.0 - jnp.eye(n)).astype(jnp.float32)


def canonical_graphs(n: int) -> jax.Array:
    """Stack of canonical adjacencies, shape ``(3, n, n)``."""
    return jnp.stack(
        [
            path_adjacency(n),
            cycle_adjacency(n),
            complete_bipartite_adjacency(n),
        ],
        axis=0,
    )


def permute_adjacency(adj: jax.Array, perm: jax.Array) -> jax.Array:
    """``B[i, j] = A[perm[i], perm[j]]``, i.e. ``B = P A P^T``."""
    return adj[perm][:, perm]


def sample_batch(
    key: jax.Array,
    batch_size: int,
    n_nodes: int = 6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draw ``(x, A*)`` pairs: one-hot class and a randomly permuted target.

    Returns
    -------
    x:
        ``(batch, n_classes)`` one-hot class vectors.
    adj_star:
        ``(batch, n, n)`` permuted target adjacencies.
    perm:
        ``(batch, n)`` permutations used to form ``A*`` from the canonical graph.
    """
    graphs = canonical_graphs(n_nodes)
    k_cls, k_perm = jax.random.split(key)
    labels = jax.random.randint(k_cls, (batch_size,), 0, N_CLASSES)
    x = jax.nn.one_hot(labels, N_CLASSES)
    perms = jax.vmap(lambda k: jax.random.permutation(k, n_nodes))(
        jax.random.split(k_perm, batch_size)
    )

    def one(label, perm):
        return permute_adjacency(graphs[label], perm)

    adj_star = jax.vmap(one)(labels, perms)
    return x, adj_star, perms
