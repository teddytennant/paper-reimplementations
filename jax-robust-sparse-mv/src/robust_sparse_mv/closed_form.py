"""Closed-form support subproblem for (P1): Propositions 1 and 2."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from robust_sparse_mv.problem import RobustSparseMV

Omega = Sequence[int] | Array


def _index(omega: Omega) -> Array:
    idx = jnp.asarray(omega, dtype=jnp.int32).reshape(-1)
    if idx.size == 0:
        raise ValueError("support omega must be nonempty")
    return idx


def submatrix(d: Array, omega: Omega) -> Array:
    idx = _index(omega)
    return d[jnp.ix_(idx, idx)]


def subvector(v: Array, omega: Omega) -> Array:
    return v[_index(omega)]


def H_omega(d: Array, r: Array, omega: Omega) -> Array:
    """Sharpe ratio ``H_ω = sqrt(r_ω^T D_ω^{-1} r_ω)`` of a nonempty support."""
    d_w = submatrix(d, omega)
    r_w = subvector(r, omega)
    inv_r = jnp.linalg.solve(d_w, r_w)
    quad = r_w @ inv_r
    return jnp.sqrt(jnp.maximum(quad, 0.0))


def xi(problem: RobustSparseMV, omega: Omega) -> Array:
    """Unique minimizer of the support-restricted problem (ZP_ω^1) (Proposition 1).

    ``ξ(ω) = (r̄ / (H_ω (H_ω − γ))) D_ω^{-1} r_ω``.
    """
    d_w = submatrix(problem.D, omega)
    r_w = subvector(problem.r, omega)
    inv_r = jnp.linalg.solve(d_w, r_w)
    h = jnp.sqrt(jnp.maximum(r_w @ inv_r, 0.0))
    scale = problem.r_bar / (h * (h - problem.gamma))
    return scale * inv_r


def Xi(problem: RobustSparseMV, omega: Omega) -> Array:
    """Zero-pad ``ξ(ω)`` to ``R^N`` (Remark 4)."""
    idx = _index(omega)
    x = jnp.zeros((problem.n,), dtype=problem.D.dtype)
    return x.at[idx].set(xi(problem, idx))


def dual_multiplier(problem: RobustSparseMV, omega: Omega) -> Array:
    """Optimal Lagrange multiplier ``μ* = 2 r̄ / (H_ω − γ)^2`` (Proposition 2)."""
    h = H_omega(problem.D, problem.r, omega)
    return 2.0 * problem.r_bar / (h - problem.gamma) ** 2


def norm_d(x: Array, d: Array) -> Array:
    """``||x||_D = sqrt(x^T D x)``."""
    return jnp.sqrt(jnp.maximum(x @ (d @ x), 0.0))


def robust_constraint_value(problem: RobustSparseMV, x: Array) -> Array:
    """Left-hand side ``r^T x − γ ||x||_D`` of the robust return constraint."""
    return problem.r @ x - problem.gamma * norm_d(x, problem.D)


def quadratic_risk(x: Array, d: Array) -> Array:
    return x @ (d @ x)


def l0_norm(x: Array, atol: float = 0.0) -> Array:
    """``||x||_0``, counting entries with ``|x_i| > atol``."""
    return jnp.sum(jnp.abs(x) > atol)


def f_beta(problem: RobustSparseMV, x: Array, atol: float = 0.0) -> Array:
    """Objective ``F_β(x) = x^T D x + β ||x||_0``."""
    return quadratic_risk(x, problem.D) + problem.beta * l0_norm(x, atol=atol)


def support_objective(problem: RobustSparseMV, omega: Omega) -> Array:
    """``F_β(Ξ(ω))`` with cardinality ``|ω|`` as used in the BnB bounds."""
    w = xi(problem, omega)
    d_w = submatrix(problem.D, omega)
    return w @ (d_w @ w) + problem.beta * w.shape[0]


def assumption2_holds(problem: RobustSparseMV) -> Array:
    """``min_i |r[i]| / sqrt(D_ii) > γ``."""
    vol = jnp.sqrt(jnp.diag(problem.D))
    return jnp.min(jnp.abs(problem.r) / vol) > problem.gamma


def singleton_ratios(problem: RobustSparseMV) -> Array:
    """``|r_i| / sqrt(D_ii)`` for each asset."""
    return jnp.abs(problem.r) / jnp.sqrt(jnp.diag(problem.D))
