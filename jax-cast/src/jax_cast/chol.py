"""CAST-CHOL approximate Cholesky (Algorithm 2) and factor application."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_cast.schur import cast_schur_tree

Array = np.ndarray | jnp.ndarray

_EDGE_EPS = 1e-15


class PivotStar(NamedTuple):
    """Recorded residual star at a pivot (one approximate-Cholesky column)."""

    pivot: int
    neighbors: np.ndarray
    conductances: np.ndarray


class CastFactor(NamedTuple):
    n: int
    rho: int
    order: np.ndarray
    stars: tuple[PivotStar, ...]


def conductance_laplacian(w: Array) -> np.ndarray:
    """Graph Laplacian L = diag(W 1) - W from a symmetric conductance matrix."""
    w = np.asarray(w, dtype=np.float64)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    deg = w.sum(axis=1)
    return np.diag(deg) - w


def _as_conductance(w: Array) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).copy()
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    w[w < 0.0] = 0.0
    return w


def _neighbors(w: np.ndarray, v: int, remaining: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = w[v]
    mask = remaining & (np.arange(w.shape[0]) != v) & (row > _EDGE_EPS)
    idx = np.flatnonzero(mask)
    return idx, row[idx]


def _min_degree_pivot(w: np.ndarray, remaining: np.ndarray) -> int:
    n = w.shape[0]
    best_v = -1
    best_deg = n + 1
    for v in range(n):
        if not remaining[v]:
            continue
        deg = int(np.count_nonzero(remaining & (np.arange(n) != v) & (w[v] > _EDGE_EPS)))
        if deg < best_deg:
            best_deg = deg
            best_v = v
    return best_v


def cast_chol(
    w: Array,
    rho: int,
    key: jax.Array,
    order: Sequence[int] | None = None,
) -> CastFactor:
    """CAST-CHOL(A, ρ) (Algorithm 2).

    ``w`` is a symmetric nonnegative off-diagonal conductance matrix
    (diagonal unused). If ``order`` is None, pivots are chosen by dynamic
    minimum degree with smallest-index tie-breaks.
    """
    residual = _as_conductance(w)
    n = residual.shape[0]
    remaining = np.ones(n, dtype=bool)
    if order is None:
        planned = None
    else:
        planned = [int(v) for v in order]
        if sorted(planned) != list(range(n)):
            raise ValueError("order must be a permutation of 0..n-1")

    stars: list[PivotStar] = []
    elim: list[int] = []
    keys = jax.random.split(key, n)

    for step in range(n):
        if planned is None:
            v = _min_degree_pivot(residual, remaining)
        else:
            v = planned[step]
            if not remaining[v]:
                raise RuntimeError(f"vertex {v} already eliminated")
        nbrs, conds = _neighbors(residual, v, remaining)
        stars.append(
            PivotStar(
                pivot=int(v),
                neighbors=np.asarray(nbrs, dtype=np.int32),
                conductances=np.asarray(conds, dtype=np.float64),
            )
        )
        elim.append(int(v))
        residual[v, :] = 0.0
        residual[:, v] = 0.0
        remaining[v] = False
        d = int(nbrs.size)
        if d <= 1:
            continue
        edges = cast_schur_tree(conds, rho, keys[step])
        for i, j, h in edges:
            ui = int(nbrs[i])
            uj = int(nbrs[j])
            residual[ui, uj] += float(h)
            residual[uj, ui] += float(h)

    return CastFactor(
        n=n,
        rho=int(rho),
        order=np.asarray(elim, dtype=np.int32),
        stars=tuple(stars),
    )


def factor_laplacian(factor: CastFactor) -> np.ndarray:
    """Approximate Laplacian C C^T assembled from recorded pivot stars."""
    n = factor.n
    c = np.zeros((n, n), dtype=np.float64)
    for star in factor.stars:
        a = float(np.sum(star.conductances))
        if a <= _EDGE_EPS:
            continue
        s = np.sqrt(a)
        c[star.pivot, star.pivot] = s
        for u, w in zip(star.neighbors, star.conductances):
            c[int(u), star.pivot] = -float(w) / s
    return c @ c.T


def _assemble_unit_ld(factor: CastFactor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = factor.n
    l = np.eye(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    pos = np.empty(n, dtype=np.int32)
    pos[factor.order] = np.arange(n, dtype=np.int32)
    for k, star in enumerate(factor.stars):
        a = float(np.sum(star.conductances))
        d[k] = a
        if a <= _EDGE_EPS:
            continue
        for u, w in zip(star.neighbors, star.conductances):
            j = int(pos[int(u)])
            l[j, k] = -float(w) / a
    return l, d, factor.order


def apply_factor(
    factor: CastFactor,
    rhs: Array,
    *,
    diag_floor: float = 1e-12,
) -> np.ndarray:
    """Solve (L D L^T) z ≈ rhs in original coordinates (unit-lower L)."""
    l, d, order = _assemble_unit_ld(factor)
    n = factor.n
    rp = np.asarray(rhs, dtype=np.float64).reshape(n)[order]
    y = np.zeros(n, dtype=np.float64)
    for k in range(n):
        s = rp[k] - float(l[k, :k] @ y[:k])
        y[k] = s
    w = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if d[k] > diag_floor:
            w[k] = y[k] / d[k]
        else:
            w[k] = 0.0
    z = np.zeros(n, dtype=np.float64)
    for k in range(n - 1, -1, -1):
        s = w[k] - float(l[k + 1 :, k] @ z[k + 1 :])
        z[k] = s
    out = np.zeros(n, dtype=np.float64)
    out[order] = z
    return out


def pcg(
    a: Array,
    b: Array,
    precon: Callable[[np.ndarray], np.ndarray],
    *,
    maxiter: int = 64,
    tol: float = 1e-8,
    project_ones: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Preconditioned CG. Residuals are relative to ||b||."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = b.size
    if project_ones:
        b = b - b.mean()

    x = np.zeros(n, dtype=np.float64)
    r = b - a @ x
    if project_ones:
        r = r - r.mean()
    z = np.asarray(precon(r), dtype=np.float64)
    if project_ones:
        z = z - z.mean()
    p = z.copy()
    rz = float(r @ z)
    bnorm = float(np.linalg.norm(b)) + 1e-30
    rel = [float(np.linalg.norm(r) / bnorm)]
    for _ in range(maxiter):
        ap = a @ p
        denom = float(p @ ap)
        if abs(denom) < 1e-30:
            break
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * ap
        if project_ones:
            r = r - r.mean()
        rel.append(float(np.linalg.norm(r) / bnorm))
        if rel[-1] < tol:
            break
        z = np.asarray(precon(r), dtype=np.float64)
        if project_ones:
            z = z - z.mean()
        rz_new = float(r @ z)
        beta = rz_new / rz if rz != 0.0 else 0.0
        p = z + beta * p
        rz = rz_new
    return x, r, rel


def factor_preconditioner(factor: CastFactor) -> Callable[[np.ndarray], np.ndarray]:
    def _solve(r: np.ndarray) -> np.ndarray:
        return apply_factor(factor, r)

    return _solve
