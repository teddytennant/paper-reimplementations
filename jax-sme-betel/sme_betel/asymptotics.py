"""The sandwich covariance behind the Bernstein-von Mises theorem.

Section 4 of arXiv:2609.01783: Theorem 1, eq (4.2), and the matching
sampling variance of the score matching estimator in eq (4.3).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = ["gamma_matrix", "delta_matrix", "sandwich_covariance"]


def gamma_matrix(g: Callable, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Gamma = E[d g(x, theta) / d theta'], Assumption 4(b), by sample average."""
    jac = jax.jit(jax.vmap(jax.jacfwd(g, argnums=1), in_axes=(0, None)))(x, theta)
    return jnp.mean(jac, axis=0)


def delta_matrix(g: Callable, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Delta = E[g(x, theta) g(x, theta)'], Assumption 4(a), by sample average."""
    gi = jax.jit(jax.vmap(g, in_axes=(0, None)))(x, theta)
    return gi.T @ gi / gi.shape[0]


def sandwich_covariance(g: Callable, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """(Gamma' Delta^{-1} Gamma)^{-1}, the limit covariance in eq (4.2).

    Theorem 1 says the SME-BETEL posterior for theta is asymptotically normal
    around theta_hat_n with covariance n^{-1} times this matrix, whether or not
    the model is correctly specified. Equation (4.3) says the same matrix is the
    asymptotic sampling covariance of sqrt(n)(theta_hat_n - theta*). With p
    moment conditions for p parameters, as here, Gamma is square and the
    expression collapses to the usual sandwich Gamma^{-1} Delta Gamma^{-T}.
    """
    gam = gamma_matrix(g, x, theta)
    dlt = delta_matrix(g, x, theta)
    return jnp.linalg.inv(gam.T @ jnp.linalg.solve(dlt, gam))
