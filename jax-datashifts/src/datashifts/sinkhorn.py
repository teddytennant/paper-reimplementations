"""Entropic optimal transport (Def. 3.1 / eq. 8) via Sinkhorn (Cuturi 2013)."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray
CostFn = Callable[[Array, Array], Array]


def as_points(x: Array) -> Array:
    """Reshape samples to (N, d)."""
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected samples of shape (N,) or (N, d), got {x.shape}")
    return x


def cost_matrix(
    x_s: Array,
    x_t: Array,
    rho: Optional[CostFn] = None,
) -> Array:
    """Pairwise cost C_ij = ρ(X_i^S, X_j^T). Euclidean by default."""
    xs = as_points(x_s)
    xt = as_points(x_t)
    if rho is not None:
        return jnp.asarray(rho(xs, xt))
    return jnp.linalg.norm(xs[:, None, :] - xt[None, :, :], axis=-1)


def _log_sinkhorn(
    cost: Array,
    beta: Array,
    n_iter: int,
) -> Tuple[Array, Array]:
    """Log-domain Sinkhorn. Returns (gamma, log_gamma)."""
    ns, nt = cost.shape
    log_a = jnp.full((ns,), -jnp.log(ns), dtype=cost.dtype)
    log_b = jnp.full((nt,), -jnp.log(nt), dtype=cost.dtype)
    log_k = -cost / beta
    log_v = jnp.zeros((nt,), dtype=cost.dtype)

    def body(_, log_v):
        log_u = log_a - jax.nn.logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - jax.nn.logsumexp(log_k.T + log_u[None, :], axis=1)
        return log_v

    log_v = lax.fori_loop(0, n_iter, body, log_v)
    log_u = log_a - jax.nn.logsumexp(log_k + log_v[None, :], axis=1)
    log_gamma = log_u[:, None] + log_k + log_v[None, :]
    gamma = jnp.exp(log_gamma)
    return gamma, log_gamma


@partial(jax.jit, static_argnames=("n_iter",))
def sinkhorn(
    cost: Array,
    beta: float = 0.01,
    n_iter: int = 2000,
) -> Tuple[Array, Array]:
    """Solve eq. 8.

    W_β = min_γ ⟨C, γ⟩ + β Σ γ_ij log γ_ij + β log(N_S N_T)
    s.t. γ 1 = (1/N_S) 1, γ^T 1 = (1/N_T) 1.

    Returns (W_β, γ*).
    """
    cost = jnp.asarray(cost)
    beta = jnp.asarray(beta, dtype=cost.dtype)
    ns, nt = cost.shape
    gamma, log_gamma = _log_sinkhorn(cost, beta, n_iter)
    transport = jnp.sum(cost * gamma)
    entropy = jnp.sum(jnp.where(gamma > 0.0, gamma * log_gamma, 0.0))
    w_beta = transport + beta * entropy + beta * jnp.log(ns * nt)
    return w_beta, gamma


def entropic_ot(
    x_s: Array,
    x_t: Array,
    beta: float = 0.01,
    rho: Optional[CostFn] = None,
    n_iter: int = 2000,
) -> Tuple[Array, Array]:
    """Plug-in W_β(μ̂, ν̂) and coupling γ* on covariate samples (eq. 8)."""
    cost = cost_matrix(x_s, x_t, rho=rho)
    return sinkhorn(cost, beta=beta, n_iter=n_iter)
