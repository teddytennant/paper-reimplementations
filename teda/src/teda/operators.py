"""Stage I linear operators: M, H_eta, P_eta, eqs (22)-(27)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# Frozen two-branch strengths from Section V.
ETA_LOW = 2.0
ETA_HIGH = 16.0


class Stage1Ops(NamedTuple):
    alpha: jnp.ndarray
    M: jnp.ndarray
    H_low: jnp.ndarray
    H_high: jnp.ndarray
    P_low: jnp.ndarray
    P_high: jnp.ndarray
    D2: jnp.ndarray
    eta_low: float
    eta_high: float


def second_difference_operator(n: int, dtype=jnp.float64) -> jnp.ndarray:
    """Second-order difference D^{(2)} of shape (n-2, n), rows [1, -2, 1]."""
    i = jnp.arange(n - 2)
    D = jnp.zeros((n - 2, n), dtype=dtype)
    D = D.at[i, i].set(jnp.asarray(1.0, dtype=dtype))
    D = D.at[i, i + 1].set(jnp.asarray(-2.0, dtype=dtype))
    D = D.at[i, i + 2].set(jnp.asarray(1.0, dtype=dtype))
    return D


def normalized_absorption_direction(alpha_tilde):
    """alpha = alpha_tilde / ||alpha_tilde||, eq (23)."""
    alpha_tilde = jnp.asarray(alpha_tilde, dtype=jnp.float64)
    return alpha_tilde / jnp.linalg.norm(alpha_tilde)


def absorption_projector(alpha):
    """M = I - alpha alpha^T, eq (24). Satisfies M alpha = 0."""
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    n = alpha.shape[0]
    return jnp.eye(n, dtype=alpha.dtype) - jnp.outer(alpha, alpha)


def branch_operators(M, eta, D2=None):
    """H_eta = A_eta^{-1} M and P_eta = I - H_eta, eqs (26)-(27).

    A_eta = M + eta D^{(2)T} D^{(2)}.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    n = M.shape[0]
    if D2 is None:
        D2 = second_difference_operator(n, dtype=M.dtype)
    DtD = D2.T @ D2
    A = M + jnp.asarray(eta, dtype=M.dtype) * DtD
    H = jnp.linalg.solve(A, M)
    P = jnp.eye(n, dtype=M.dtype) - H
    return H, P, A, D2


def build_stage1_operators(
    alpha_tilde,
    eta_low: float = ETA_LOW,
    eta_high: float = ETA_HIGH,
) -> Stage1Ops:
    """Precompute both frozen-eta branches from the integrated absorption vector."""
    alpha = normalized_absorption_direction(alpha_tilde)
    M = absorption_projector(alpha)
    D2 = second_difference_operator(alpha.shape[0], dtype=alpha.dtype)
    H_low, P_low, _, _ = branch_operators(M, eta_low, D2=D2)
    H_high, P_high, _, _ = branch_operators(M, eta_high, D2=D2)
    return Stage1Ops(
        alpha=alpha,
        M=M,
        H_low=H_low,
        H_high=H_high,
        P_low=P_low,
        P_high=P_high,
        D2=D2,
        eta_low=float(eta_low),
        eta_high=float(eta_high),
    )
