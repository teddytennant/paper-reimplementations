"""Direct Spectral Laplacian Estimation (arXiv:2609.23668).

Algorithms 1--3: DSLE, MDS+, and sparsify. Data matrix X is n by p,
rows are samples and columns are nodes. Columns must be pairwise distinct.
The paper's rank guarantee needs gamma > 1; gamma = 1 is the resistance
identity that the estimator inverts.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import Array

_DTYPE = jnp.float64


def centering_matrix(p: int) -> Array:
    """J = I - 11^T / p."""
    eye = jnp.eye(p, dtype=_DTYPE)
    return eye - jnp.ones((p, p), dtype=_DTYPE) / p


def sample_covariance(X: Array) -> Array:
    """S = (1/n) X^T X. Divide by n, not n-1. No column centering."""
    X = jnp.asarray(X, dtype=_DTYPE)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    n = X.shape[0]
    if n < 1:
        raise ValueError("X must contain at least one sample")
    return (X.T @ X) / n


def resistance_from_covariance(S: Array) -> Array:
    """R_ij = S_ii + S_jj - 2 S_ij = (1/n) ||column_i - column_j||^2.

    Symmetric, zero diagonal, and entrywise non-negative up to rounding.
    """
    S = _sym(jnp.asarray(S, dtype=_DTYPE))
    diag = jnp.diag(S)
    R = diag[:, None] + diag[None, :] - 2.0 * S
    return _zero_diag(_sym(R))


def mds_plus(Y: Array) -> Array:
    """Algorithm 2, MDS+.

    Project the centered Gram matrix of a symmetric zero-diagonal Y onto
    the positive semidefinite cone and return the induced distance matrix.
    Negative eigenvalues of B = -1/2 J Y J are set to 0.
    """
    Y = _zero_diag(_sym(jnp.asarray(Y, dtype=_DTYPE)))
    p = Y.shape[0]
    J = centering_matrix(p)
    B = _sym(-0.5 * (J @ Y @ J))
    evals, evecs = jnp.linalg.eigh(B)
    lam_plus = jnp.maximum(evals, 0.0)
    G = _sym((evecs * lam_plus) @ evecs.T)
    gdiag = jnp.diag(G)
    D = gdiag[:, None] + gdiag[None, :] - 2.0 * G
    return _zero_diag(_sym(D))


def laplacian_from_resistance(R: Array) -> Array:
    """Stable inverse L = (-1/2 J R J)^+.

    L = (A + 11^T / p)^{-1} - 11^T / p, with A = -1/2 J R J.
    A solve is used in place of an explicit inverse.
    """
    R = _sym(jnp.asarray(R, dtype=_DTYPE))
    p = R.shape[0]
    J = centering_matrix(p)
    A = _sym(-0.5 * (J @ R @ J))
    P = jnp.ones((p, p), dtype=_DTYPE) / p
    M = _sym(A + P)
    L = jnp.linalg.solve(M, jnp.eye(p, dtype=_DTYPE)) - P
    return _sym(L)


def regularized_resistance(R_hat: Array, gamma: float) -> Array:
    """R_bar = MDS+(R_hat^{∘γ})^{∘(1/γ)}."""
    gamma = _as_gamma(gamma)
    R_hat = _zero_diag(_sym(jnp.asarray(R_hat, dtype=_DTYPE)))
    # Squared distances are nonnegative. A fractional power is undefined
    # for the ~1e-16 negatives left by cancellation, so clip those only.
    powered = jnp.power(jnp.maximum(R_hat, 0.0), gamma)
    D = mds_plus(powered)
    R_bar = jnp.power(jnp.maximum(D, 0.0), 1.0 / gamma)
    return _zero_diag(_sym(R_bar))


def laplacian_from_covariance(S: Array, gamma: float) -> Array:
    """Algorithm 1 after the covariance is formed."""
    R_hat = resistance_from_covariance(S)
    R_bar = regularized_resistance(R_hat, gamma)
    return laplacian_from_resistance(R_bar)


def dsle(X: Array, gamma: float) -> Array:
    """Algorithm 1, DSLE(X, γ).

    Returns the dense signed Laplacian L_bar. Preconditions from the paper:
    gamma > 1 and pairwise distinct columns. gamma = 1 is accepted because
    it is the exact resistance inverse.
    """
    X = jnp.asarray(X, dtype=_DTYPE)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    n, p = X.shape
    if n < 1 or p < 2:
        raise ValueError("need at least one sample and two nodes")
    return laplacian_from_covariance(sample_covariance(X), gamma)


def maximum_spanning_tree(W: Array) -> Array:
    """Boolean mask of a maximum spanning tree (Kruskal).

    Every upper-triangle edge is eligible, including negative weights, so
    the tree is connected whenever the complete graph on those edges is.
    Ties break by smaller endpoint index i, then j. The mask is symmetric
    with a zero diagonal.
    """
    W_np = np.asarray(W, dtype=np.float64)
    W_np = 0.5 * (W_np + W_np.T)
    np.fill_diagonal(W_np, 0.0)
    p = W_np.shape[0]
    ii, jj = np.triu_indices(p, k=1)
    weights = W_np[ii, jj]
    # lexsort primary key is last: largest weight, then smaller i, then j.
    order = np.lexsort((jj, ii, -weights))
    parent = np.arange(p, dtype=np.int32)
    rank = np.zeros(p, dtype=np.int32)
    mask = np.zeros((p, p), dtype=bool)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    n_added = 0
    for t in order:
        i = int(ii[t])
        j = int(jj[t])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1
        mask[i, j] = True
        mask[j, i] = True
        n_added += 1
        if n_added == p - 1:
            break
    return jnp.asarray(mask)


def sparsify(L_bar: Array, S: Array, M: int) -> list[Array]:
    """Algorithm 3, Sparsify(L_bar, S, M).

    For m = 1..M, keep the maximum spanning tree of W together with edges
    whose weight exceeds tau = max(W) * m / M, form the combinatorial
    Laplacian, and rescale it so that trace(L_m S) = p - 1.
    """
    M = int(M)
    if M < 1:
        raise ValueError("M must be at least 1")
    L_bar = _sym(jnp.asarray(L_bar, dtype=_DTYPE))
    S = _sym(jnp.asarray(S, dtype=_DTYPE))
    p = L_bar.shape[0]
    if S.shape != (p, p):
        raise ValueError("S must be p by p, matching L_bar")
    W = -L_bar
    W = _zero_diag(_sym(W))
    mst = maximum_spanning_tree(W)
    # Diagonal zeros make max(W) >= 0, so every threshold is non-negative,
    # matching the paper's sweep.
    w_max = jnp.max(W)
    eye = jnp.eye(p, dtype=bool)
    out: list[Array] = []
    for m in range(1, M + 1):
        tau = w_max * (m / M)
        keep = (mst | (W > tau)) & ~eye
        keep = keep | jnp.swapaxes(keep, 0, 1)
        W_m = _zero_diag(jnp.where(keep, W, 0.0))
        W_m = _sym(W_m)
        L_m = _combinatorial_laplacian(W_m)
        tr = jnp.trace(L_m @ S)
        L_m = ((p - 1) / tr) * L_m
        out.append(_sym(L_m))
    return out


def _combinatorial_laplacian(W: Array) -> Array:
    """L_ii = sum_j W_ij, L_ij = -W_ij. W is symmetric with zero diagonal."""
    deg = jnp.sum(W, axis=1)
    L = -W
    return L.at[jnp.diag_indices(W.shape[0])].set(deg)


def _as_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not gamma > 0.0:
        raise ValueError("gamma must be positive")
    return gamma


def _sym(A: Array) -> Array:
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _zero_diag(A: Array) -> Array:
    return A.at[jnp.diag_indices(A.shape[-1])].set(0.0)
