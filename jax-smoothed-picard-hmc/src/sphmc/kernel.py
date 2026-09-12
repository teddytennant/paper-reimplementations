"""Stochastic smoothed score (3.2) and Picard HMC step (Algorithm 3.1)."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sphmc.quadrature import ChebyshevLobatto


ScoreFn = Callable[[Array, Array], Array]


class PicardState(NamedTuple):
    """Nodes and endpoint after a Picard HMC step."""

    x: Array
    p: Array
    x_nodes: Array  # (J, d)
    p_nodes: Array  # (J, d)


def stochastic_score(
    y: Array,
    G: Array,
    grad_fn: Callable[[Array], Array],
    prox_fn: Callable[[Array], Array],
    eta: float,
) -> Array:
    """ĝ_η(y; G) := ∇V(prox_{η V}(y) + √η G)."""
    return grad_fn(prox_fn(y) + jnp.sqrt(eta) * G)


def make_stochastic_score(
    grad_fn: Callable[[Array], Array],
    prox_fn: Callable[[Array], Array],
    eta: float,
) -> ScoreFn:
    """Bind ∇V, prox, and η into a score_fn(y, G)."""

    def score_fn(y: Array, G: Array) -> Array:
        return stochastic_score(y, G, grad_fn, prox_fn, eta)

    return score_fn


def make_quadratic_score(A: ArrayLike, eta: float) -> ScoreFn:
    """ĝ_η for V(x) = (1/2) x^T A x, using the closed-form prox."""
    A = jnp.asarray(A)
    d = A.shape[0]
    eye = jnp.eye(d, dtype=A.dtype)
    system = eye + eta * A
    sqrt_eta = jnp.sqrt(eta)

    def score_fn(y: Array, G: Array) -> Array:
        x_prox = jnp.linalg.solve(system, y)
        return A @ (x_prox + sqrt_eta * G)

    return score_fn


def make_exact_smoothed_quadratic_score(A: ArrayLike, eta: float) -> ScoreFn:
    """True ∇V_η for quadratic V: π_η = N(0, A^{-1} + η I)."""
    A = jnp.asarray(A)
    d = A.shape[0]
    cov = jnp.linalg.inv(A) + eta * jnp.eye(d, dtype=A.dtype)
    precision = jnp.linalg.inv(cov)

    def score_fn(y: Array, G: Array) -> Array:
        del G
        return precision @ y

    return score_fn


def picard_hmc_step(
    key: Array,
    x: Array,
    p: Array,
    score_fn: ScoreFn,
    quad: ChebyshevLobatto,
    n_picard: int,
) -> PicardState:
    """One Algorithm 3.1 step.

    Draws G_1..G_J once and reuses them across Picard iterations.
    Output is the last Chebyshev–Lobatto node after ``n_picard`` updates.
    """
    x = jnp.asarray(x)
    p = jnp.asarray(p)
    dtype = x.dtype
    t = jnp.asarray(quad.t, dtype=dtype)
    omega = jnp.asarray(quad.omega, dtype=dtype)
    omega_prime = jnp.asarray(quad.omega_prime, dtype=dtype)
    J = t.shape[0]
    d = x.shape[-1]
    G = jax.random.normal(key, (J, d), dtype=dtype)

    x_nodes = x[None, :] + t[:, None] * p[None, :]
    p_nodes = jnp.broadcast_to(p[None, :], (J, d))

    def body(_n, carry):
        x_cur, _p_cur = carry
        g = jax.vmap(score_fn)(x_cur, G)
        x_new = x[None, :] + t[:, None] * p[None, :] - omega @ g
        p_new = p[None, :] - omega_prime @ g
        return x_new, p_new

    x_nodes, p_nodes = jax.lax.fori_loop(
        0, n_picard, body, (x_nodes, p_nodes)
    )
    return PicardState(
        x=x_nodes[-1],
        p=p_nodes[-1],
        x_nodes=x_nodes,
        p_nodes=p_nodes,
    )
