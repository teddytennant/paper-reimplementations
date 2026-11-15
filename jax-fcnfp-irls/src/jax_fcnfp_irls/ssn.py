"""Dual semismooth Newton solver for the weighted flow QP (Sec. III-C).

The Newton system is a weighted graph Laplacian on the free subgraph.
This small repo solves it with a dense linear solve plus a tiny ridge;
there is no approx-Cholesky / Spielman-Teng PCG and no IPM fallback.
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_fcnfp_irls.network import conservation_residual

MAX_NEWTON = 40
RIDGE = 1e-8
FREE_ATOL = 0.0


def _as_f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def primal_from_dual(y, C, w, c):
    """f(y) = clip(W^{-1} C^T y, 0, c) componentwise."""
    y = _as_f64(y)
    C = _as_f64(C)
    w = _as_f64(w)
    c = _as_f64(c)
    reduced = C.T @ y
    return jnp.clip(reduced / w, 0.0, c)


def free_arc_mask(y, C, w, c, atol: float = FREE_ATOL):
    """Free arcs: 0 < (C^T y)_e < w_e c_e."""
    y = _as_f64(y)
    C = _as_f64(C)
    w = _as_f64(w)
    c = _as_f64(c)
    reduced = C.T @ y
    return (reduced > atol) & (reduced < w * c - atol)


def newton_diag(y, C, w, c, atol: float = FREE_ATOL):
    """D(y) = diag(w^{-1} * 1{0 < (C^T y)_e < w_e c_e})."""
    w = _as_f64(w)
    mask = free_arc_mask(y, C, w, c, atol=atol)
    return (1.0 / w) * mask.astype(jnp.float64)


def newton_matrix(C, D_diag):
    """Weighted Laplacian C D C^T."""
    C = _as_f64(C)
    D_diag = _as_f64(D_diag)
    return C @ (D_diag[:, None] * C.T)


def _solve_laplacian(L, rhs, ridge: float = RIDGE):
    """Dense solve of a graph Laplacian with ridge and mean-centering (gauge)."""
    L = _as_f64(L)
    rhs = _as_f64(rhs)
    n = L.shape[0]
    Lreg = L + ridge * jnp.eye(n, dtype=jnp.float64)
    rhs_c = rhs - jnp.mean(rhs)
    y, *_ = jnp.linalg.lstsq(Lreg, rhs_c, rcond=None)
    return y - jnp.mean(y)


def predictor_y(C, w, f=None, d=None, c=None, ridge: float = RIDGE):
    """Dual predictor: Laplacian solve with D = W^{-1} (all arcs, or free arcs).

    Matches C^T y ≈ W f on those arcs, i.e. (C D C^T) y ≈ C f (or d).
    Does not start Newton at y = 0 (where D(0)=0 is degenerate).
    """
    C = _as_f64(C)
    w = _as_f64(w)
    n = C.shape[0]
    D = 1.0 / w
    if f is not None and c is not None:
        f = _as_f64(f)
        c = _as_f64(c)
        free = (f > 1e-10) & (f < c - 1e-10)
        if jnp.any(free):
            D = jnp.where(free, 1.0 / w, 0.0)
        rhs = C @ f
    elif d is not None:
        rhs = _as_f64(d)
    else:
        rhs = jnp.zeros(n, dtype=jnp.float64)
    L = newton_matrix(C, D)
    return _solve_laplacian(L, rhs, ridge=ridge)


def dual_ssn(
    C,
    d,
    c,
    w,
    y0=None,
    f_current=None,
    max_newton: int = MAX_NEWTON,
    ridge: float = RIDGE,
    tol: float = 1e-10,
):
    """Solve min_{f in F} (1/2) sum w_e f_e^2 by dual semismooth Newton.

    F = {f : C f = d, 0 <= f <= c}. Cap at `max_newton` Newton iterations.
    Returns (f, y, residual_norm, n_iters).
    """
    C = _as_f64(C)
    d = _as_f64(d)
    c = _as_f64(c)
    w = _as_f64(w)
    w = jnp.clip(w, 1e-12, 1e12)

    if y0 is not None:
        y = _as_f64(y0)
        y = y - jnp.mean(y)
    else:
        y = predictor_y(C, w, f=f_current, d=d, c=c, ridge=ridge)

    eta = 0.0
    f = primal_from_dual(y, C, w, c)
    R = d - C @ f
    best_f = f
    best_y = y
    best_res = float(jnp.linalg.norm(R, ord=jnp.inf))
    n_iters = 0

    for k in range(max_newton):
        n_iters = k + 1
        f = primal_from_dual(y, C, w, c)
        R = d - C @ f
        rinf = float(jnp.linalg.norm(R, ord=jnp.inf))
        if rinf < best_res:
            best_res = rinf
            best_f = f
            best_y = y
        if rinf < tol:
            break

        D = newton_diag(y, C, w, c)
        if eta > 0.0:
            D = D + eta / w
        elif float(jnp.sum(D)) <= 0.0:
            D = D + 1e-2 / w

        L = newton_matrix(C, D)
        dy = _solve_laplacian(L, R, ridge=ridge)

        step = 1.0
        accepted = False
        r2 = float(jnp.linalg.norm(R))
        for _ in range(12):
            y_try = y + step * dy
            y_try = y_try - jnp.mean(y_try)
            f_try = primal_from_dual(y_try, C, w, c)
            R_try = d - C @ f_try
            r2_try = float(jnp.linalg.norm(R_try))
            if r2_try <= (1.0 - 1e-4 * step) * r2 + 1e-16:
                y = y_try
                accepted = True
                break
            step *= 0.5
        if not accepted:
            eta = 1e-2 if eta <= 0.0 else min(eta * 10.0, 1e2)
            y = y + 0.1 * dy
            y = y - jnp.mean(y)

    f = primal_from_dual(y, C, w, c)
    R = d - C @ f
    rinf = float(jnp.linalg.norm(R, ord=jnp.inf))
    if rinf < best_res:
        best_res = rinf
        best_f = f
        best_y = y

    best_f = jnp.clip(best_f, 0.0, c)
    return best_f, best_y, best_res, n_iters


def residual_inf(C, f, d):
    return float(jnp.linalg.norm(conservation_residual(C, f, d), ord=jnp.inf))
