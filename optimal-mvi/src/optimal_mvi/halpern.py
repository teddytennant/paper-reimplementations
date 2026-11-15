"""Inexact accelerated Halpern iteration (Algorithm 1), unconstrained.

Equation (8), integer p >= 1:

    q = (3p - 1) / 2
    η_0 = 1 / (L_p D^{p-1})
    β_t = q / (t + q)                         for t >= 1
    η_t = η_{t-1} (t + 2q - 1) / (t + q)      for t >= 1
    δ_T = D / T^{3/2}

The resolvent subproblem uses G_t(z) = F(z) + η_t^{-1} (z - x_t), whose
strong-monotonicity constant is η_t^{-1}. Proj_X is the identity on R^d,
and v_t = F(u_t) because the normal cone is {0}.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from optimal_mvi.tensor import r_adaptive_atm


def parameter_schedule(p, L_p, D, T):
    """Step-size schedule (8). Returns q, eta[0:T], beta[0:T], delta_T.

    beta[0] is unused; equation (8) defines β_t for t >= 1.
    """
    p = int(p)
    T = int(T)
    if p < 1:
        raise ValueError("p must be a positive integer")
    if T < 1:
        raise ValueError("T must be at least 1")
    L_p = float(L_p)
    D = float(D)
    if L_p <= 0 or D < 0:
        raise ValueError("L_p must be positive and D nonnegative")
    q = (3.0 * p - 1.0) / 2.0
    eta = [0.0] * T
    beta = [float("nan")] * T
    eta[0] = 1.0 / (L_p * D ** (p - 1))
    for t in range(1, T):
        beta[t] = q / (t + q)
        eta[t] = eta[t - 1] * (t + 2.0 * q - 1.0) / (t + q)
    delta_T = D / (T ** 1.5)
    return q, eta, beta, delta_T


@dataclass
class HalpernResult:
    u: jnp.ndarray
    v: jnp.ndarray
    residuals: jnp.ndarray
    accepted: bool
    inner_accepted: list


def inexact_accelerated_halpern(F, x0, T, L_p, D, M, p=2):
    """Algorithm 1 on X = R^d. Only p=2 calls the tensor subroutine.

    Returns u_{T-1}, v_{T-1}, the trajectory ||v_t||, and whether every
    inner R-Adaptive-ATM call met (1/μ) ||G_t(u)|| ≤ δ_T.
    """
    if int(p) != 2:
        raise NotImplementedError("shipped solver calls the tensor step only at p=2")
    x0 = jnp.asarray(x0, dtype=jnp.float64).reshape(-1)
    q, eta, beta, delta_T = parameter_schedule(p, L_p, D, T)
    M = float(M)

    def anchored(mu, point):
        point = jnp.asarray(point, dtype=jnp.float64)

        def G(z):
            z = jnp.asarray(z, dtype=jnp.float64)
            return F(z) + mu * (z - point)

        return G

    residuals = []
    inner_accepted = []

    # t = 0
    y = x0
    mu = 1.0 / eta[0]
    G = anchored(mu, x0)
    inner = r_adaptive_atm(G, y, mu, M, delta_T, p=2)
    u = inner.u
    n = inner.n
    v = F(u) + n
    residuals.append(float(jnp.linalg.norm(v)))
    inner_accepted.append(bool(inner.accepted))

    for t in range(1, int(T)):
        x_t = (1.0 - beta[t]) * u + beta[t] * x0
        y_t = x_t - eta[t] * v
        mu = 1.0 / eta[t]
        G = anchored(mu, x_t)
        inner = r_adaptive_atm(G, y_t, mu, M, delta_T, p=2)
        u = inner.u
        n = inner.n
        v = F(u) + n
        residuals.append(float(jnp.linalg.norm(v)))
        inner_accepted.append(bool(inner.accepted))

    accepted = all(inner_accepted)
    return HalpernResult(
        u=u,
        v=v,
        residuals=jnp.asarray(residuals, dtype=jnp.float64),
        accepted=accepted,
        inner_accepted=inner_accepted,
    )
