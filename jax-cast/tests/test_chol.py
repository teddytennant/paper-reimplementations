"""CAST-CHOL tests on a path Laplacian."""

from __future__ import annotations

import jax
import numpy as np

from jax_cast.chol import (
    apply_factor,
    cast_chol,
    conductance_laplacian,
    factor_laplacian,
    factor_preconditioner,
    pcg,
)


def _path_conductance(n: int, w: float = 1.0) -> np.ndarray:
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        c[i, i + 1] = w
        c[i + 1, i] = w
    return c


def test_cast_chol_path_factor_and_kernel():
    n = 6
    w = _path_conductance(n)
    a = conductance_laplacian(w)
    factor = cast_chol(w, rho=1, key=jax.random.PRNGKey(0))
    assert len(factor.stars) in (n - 1, n)
    assert factor.order.size == n
    assert set(factor.order.tolist()) == set(range(n))

    lhat = factor_laplacian(factor)
    evals = np.sort(np.linalg.eigvalsh(lhat))
    assert evals[0] >= -1e-8
    assert evals[1] > 1e-8
    ones = np.ones(n)
    assert np.allclose(lhat @ ones, 0.0, atol=1e-8)

    b = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
    b = b - b.mean()
    precon = factor_preconditioner(factor)
    _x_p, r_p, rel_p = pcg(a, b, precon, maxiter=1, tol=0.0)
    _x_u, r_u, rel_u = pcg(a, b, lambda r: r, maxiter=1, tol=0.0)
    assert rel_p[-1] < rel_u[-1]
    assert np.linalg.norm(r_p) < np.linalg.norm(r_u)


def test_cast_chol_explicit_order():
    n = 5
    w = _path_conductance(n)
    order = list(range(n))
    factor = cast_chol(w, rho=1, key=jax.random.PRNGKey(1), order=order)
    assert factor.order.tolist() == order
    lhat = factor_laplacian(factor)
    a = conductance_laplacian(w)
    # Path + CAST-1 is exact under endpoint peeling and under 0..n-1 order
    # (each interior pivot has degree 2, unique tree).
    assert np.allclose(lhat, a, atol=1e-8)


def test_apply_factor_matches_operator():
    w = _path_conductance(6)
    a = conductance_laplacian(w)
    factor = cast_chol(w, rho=1, key=jax.random.PRNGKey(2))
    lhat = factor_laplacian(factor)
    b = np.arange(6, dtype=np.float64)
    b = b - b.mean()
    z = apply_factor(factor, b)
    # Lhat z ≈ b on 1-perp for an exact path factor.
    resid = a @ z - b
    resid = resid - resid.mean()
    assert np.linalg.norm(resid) < 1e-6
    _ = lhat
