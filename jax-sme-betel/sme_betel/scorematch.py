"""Hyvarinen score matching objective and its first-order conditions.

Section 2.1 of arXiv:2609.01783, equations (2.1) through (2.4).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "score_integrand",
    "sm_objective",
    "moment_fn",
    "moment_matrix",
    "sm_estimator",
    "newton_minimize",
]

LogQ = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def score_integrand(log_q: LogQ, x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """The per-observation summand of the sample score matching loss, eq (2.3).

    Returns tr(hess_x log q) + 0.5 ||grad_x log q||^2. Because log p and log q
    differ by a factor that does not depend on x, the normalizing constant
    Z(theta) never enters, which is the point of eq (2.2).

    Args:
        log_q: scalar log unnormalized density, called as log_q(x, theta).
        x: a single observation, shape (m,).
        theta: parameter vector, shape (p,).
    """
    grad_x = jax.grad(log_q, argnums=0)(x, theta)
    hess_x = jax.jacfwd(jax.grad(log_q, argnums=0), argnums=0)(x, theta)
    return jnp.trace(hess_x) + 0.5 * jnp.dot(grad_x, grad_x)


def sm_objective(log_q: LogQ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build Jhat(theta; x_1:n), the sample score matching loss of eq (2.3).

    The additive constant of eq (2.3), which does not depend on theta, is dropped.
    """

    per_obs = jax.vmap(lambda x, theta: score_integrand(log_q, x, theta), in_axes=(0, None))

    def jhat(theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(per_obs(x, theta))

    return jhat


def moment_fn(log_q: LogQ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build g(x, theta), the moment function of eq (2.4).

    g is the theta-gradient of the integrand in eq (2.3), so the population
    condition E_p[g(x, theta*)] = 0 is the first-order condition of the score
    matching problem (2.1). Everything is exact autodiff, no finite differences.
    """

    def g(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda th: score_integrand(log_q, x, th))(theta)

    return g


def moment_matrix(g: Callable) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Stack g(x_i, theta) into the n by p matrix that ETEL consumes."""
    batched = jax.vmap(g, in_axes=(0, None))

    def gmat(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return batched(x, theta)

    return gmat


def newton_minimize(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    theta_init: jnp.ndarray,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> jnp.ndarray:
    """Damped Newton with backtracking, for smooth low-dimensional objectives."""
    f = jax.jit(f)
    grad_f = jax.jit(jax.grad(f))
    hess_f = jax.jit(jax.hessian(f))

    theta = jnp.asarray(theta_init, dtype=jnp.float64)
    p = theta.shape[0]
    for _ in range(max_iter):
        gradient = grad_f(theta)
        if float(jnp.max(jnp.abs(gradient))) < tol:
            break
        hessian = hess_f(theta)
        # Levenberg style damping so a nearly singular Hessian still gives a
        # usable direction.
        ridge = 1e-10 * jnp.maximum(1.0, jnp.max(jnp.abs(hessian)))
        step = jnp.linalg.solve(hessian + ridge * jnp.eye(p), gradient)
        if float(jnp.dot(step, gradient)) <= 0:
            step = gradient
        f0 = float(f(theta))
        alpha = 1.0
        for _ in range(50):
            if float(f(theta - alpha * step)) < f0:
                break
            alpha *= 0.5
        else:
            break
        theta = theta - alpha * step
    return theta


def sm_estimator(
    jhat: Callable,
    x: jnp.ndarray,
    theta_init: jnp.ndarray,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> jnp.ndarray:
    """Minimize Jhat of eq (2.3), giving theta_hat_n as used in Theorem 1."""
    return newton_minimize(lambda th: jhat(th, x), theta_init, max_iter, tol)
