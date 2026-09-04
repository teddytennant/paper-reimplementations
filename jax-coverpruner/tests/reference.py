"""Plain numpy restatement of the paper's equations, written independently of the
package so the tests compare two implementations rather than one with itself."""

import itertools

import numpy as np


def utilities(V):
    """A_ij = max(0, M_ij - mean_k M_ik) with M cosine similarity (Eqs. 2, 3)."""
    V = np.asarray(V, dtype=np.float64)
    M = (V / np.linalg.norm(V, axis=1, keepdims=True)) @ (
        V / np.linalg.norm(V, axis=1, keepdims=True)
    ).T
    return np.maximum(0.0, M - M.mean(axis=1, keepdims=True))


def F(A, w, S):
    """F(S) = sum_i w_i max_{j in S} A_ij (Eq. 5). F(empty) = 0."""
    S = list(S)
    if not S:
        return 0.0
    return float((np.asarray(w) * np.asarray(A)[:, S].max(axis=1)).sum())


def gain(A, w, S, e):
    """Marginal gain F(S + e) - F(S)."""
    return F(A, w, list(S) + [e]) - F(A, w, S)


def brute_force(A, w, m):
    """Exact argmax of F over all subsets of size m. Only for tiny n."""
    n = A.shape[0]
    best, best_S = -np.inf, None
    for S in itertools.combinations(range(n), m):
        v = F(A, w, S)
        if v > best:
            best, best_S = v, S
    return best, best_S


def random_instance(seed, n=12, d=8, clusters=4, spread=0.5, weighted=True):
    """Clustered tokens plus demand weights, the shape real instances have."""
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(clusters, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    grp = rng.integers(0, clusters, size=n)
    V = C[grp] + spread * rng.normal(size=(n, d))
    w = np.exp(0.3 * rng.random(n)) if weighted else np.ones(n)
    return utilities(V), w, V, grp
