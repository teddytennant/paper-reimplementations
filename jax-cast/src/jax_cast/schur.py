"""CAST-SCHURTREE: spanning-tree Schur-clique replacement (Algorithm 1)."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_cast.prufer import prufer_decode

Array = np.ndarray | jnp.ndarray


def exact_schur_clique(a: Array) -> np.ndarray:
    """Exact Schur clique Laplacian K_v = diag(a) - a a^T / sum(a).

    Equivalently, undirected edge (i, j) has weight a_i a_j / a.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    total = float(a.sum())
    if a.size == 0 or total <= 0.0:
        return np.zeros((a.size, a.size), dtype=np.float64)
    return np.diag(a) - np.outer(a, a) / total


def leverage_scores(a: Array) -> np.ndarray:
    """CAST-1 tree-inclusion probabilities τ_ij = (a_i + a_j) / sum(a)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    total = float(a.sum())
    d = a.size
    tau = np.zeros((d, d), dtype=np.float64)
    if total <= 0.0:
        return tau
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = (a[i] + a[j]) / total
    return tau


def inverse_marginal_weight(ai: float, aj: float, rho: int = 1) -> float:
    """Contracted CAST-ρ conductance a_i a_j / (ρ (a_i + a_j))."""
    return float(ai) * float(aj) / (float(rho) * (float(ai) + float(aj)))


def laplacian_from_edges(
    d: int, edges: Sequence[tuple[int, int, float]]
) -> np.ndarray:
    """Assemble a d×d Laplacian from an undirected weighted edge list."""
    k = np.zeros((d, d), dtype=np.float64)
    for i, j, h in edges:
        if i == j:
            continue
        w = float(h)
        k[i, i] += w
        k[j, j] += w
        k[i, j] -= w
        k[j, i] -= w
    return k


def _terminals_connected(d: int, edges: Sequence[tuple[int, int, float]]) -> bool:
    if d <= 1:
        return True
    parent = list(range(d))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _h in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj
    roots = {find(i) for i in range(d)}
    return len(roots) == 1


def is_connected(d: int, edges: Sequence[tuple[int, int, float]]) -> bool:
    return _terminals_connected(d, edges)


def cast_schur_tree(
    a: Array,
    rho: int,
    key: jax.Array,
) -> list[tuple[int, int, float]]:
    """Sample CAST-SCHURTREE(a_1..a_d, ρ) (Algorithm 1).

    Returns a multiset of terminal edges (i, j, h) with 0-based neighbor
    indices matching ``a``. Auxiliary copies are labeled
    p = i * ρ + k for terminal i and copy k ∈ {0, …, ρ-1}.
    """
    if rho < 1:
        raise ValueError("rho must be a positive integer")
    a = jnp.asarray(a, dtype=jnp.float32).reshape(-1)
    d = int(a.shape[0])
    if d <= 1:
        return []

    total = jnp.sum(a)
    pi = a / total
    m = int(rho) * d
    n_symbols = m - 2
    key_i, key_k = jax.random.split(key)
    if n_symbols > 0:
        terminals = jax.random.choice(key_i, d, shape=(n_symbols,), p=pi)
        copies = jax.random.randint(key_k, (n_symbols,), 0, int(rho))
        code = np.asarray(terminals * int(rho) + copies, dtype=int).tolist()
    else:
        code = []

    aux_vertices = list(range(m))
    aux_edges = prufer_decode(code, aux_vertices)
    a_np = np.asarray(a, dtype=np.float64)
    out: list[tuple[int, int, float]] = []
    for p, q in aux_edges:
        i = int(p) // int(rho)
        j = int(q) // int(rho)
        if i == j:
            continue
        h = inverse_marginal_weight(float(a_np[i]), float(a_np[j]), int(rho))
        if i > j:
            i, j = j, i
        out.append((i, j, h))
    return out
