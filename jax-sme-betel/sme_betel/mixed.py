"""Mixed-domain score matching for observations x = (y, s) with s compact.

Section 2.3 of arXiv:2609.01783: the objective (2.8), its computable form in
Proposition 1 / eq (2.9), and the boundary weight of Appendix S3.1, eq (S1).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .scorematch import newton_minimize

__all__ = [
    "smoothstep",
    "boundary_weight_1d",
    "box_weight",
    "mixed_integrand",
    "mixed_objective",
    "mixed_moment_fn",
    "mixed_estimator",
]


def smoothstep(u: jnp.ndarray) -> jnp.ndarray:
    """The cubic 3u^2 - 2u^3, clamped to [0, 1] outside the taper."""
    u = jnp.clip(u, 0.0, 1.0)
    return 3.0 * u**2 - 2.0 * u**3


def boundary_weight_1d(x: jnp.ndarray, a: float, b: float, eps: float) -> jnp.ndarray:
    """The weight function h of eq (S1) on the interval [a, b].

    h is zero at both endpoints, rises as the cubic 3t^2 - 2t^3 in t = (x-a)/eps,
    equals one on [a+eps, b-eps], and tapers back down over the last eps. It is
    continuously differentiable, since the cubic has zero slope at t = 0 and
    t = 1. The paper uses eps = 0.2 in every numerical study.

    Written as a product of the two one-sided tapers, which agrees with eq (S1)
    whenever b - a >= 2 eps. For a narrower interval the two tapers overlap and
    h stays below one in the middle; it is still a valid weight, just not the
    exact piecewise formula.
    """
    return smoothstep((x - a) / eps) * smoothstep((b - x) / eps)


def box_weight(lower, upper, eps: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """h(s) = prod_j h_j(s_j) on a rectangle, the multivariate form of eq (S1)."""
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    def h(s: jnp.ndarray) -> jnp.ndarray:
        return jnp.prod(boundary_weight_1d(s, lower, upper, eps))

    return h


def mixed_integrand(
    log_q: Callable, weight: Callable, y: jnp.ndarray, s: jnp.ndarray, theta: jnp.ndarray
) -> jnp.ndarray:
    """The per-observation summand of J_MD in eq (2.9).

    Equation (2.9) reads

        0.5 h(s) ||psi||^2 + h(s) tr(hess_y log p(y|s)) + h(s) Laplacian_s log p
        + grad_s h(s)' grad_s log p.

    Since log p(s; theta) does not depend on y, tr(hess_y log p(y|s)) equals
    tr(hess_y log p(y, s)), so the two middle terms are the full Laplacian of
    the joint log density in (y, s) times h(s). That is what is computed here.

    Only derivatives in (y, s) appear, so log_q may be the joint density up to
    any factor depending on theta alone. In particular the spatial normalizing
    integral of p(s; theta) = exp{xi(s; theta)} / int exp{xi} drops out, which
    is the point made after Proposition 1.

    Args:
        log_q: scalar, called as log_q(y, s, theta).
        weight: the boundary weight h, called as weight(s).
        y: Euclidean component, shape (d,).
        s: compact-domain component, shape (k,).
    """
    d = y.shape[0]
    flat = jnp.concatenate([y, s])

    def f(z):
        return log_q(z[:d], z[d:], theta)

    grad = jax.grad(f)(flat)
    hess = jax.jacfwd(jax.grad(f))(flat)
    h = weight(s)
    grad_h = jax.grad(weight)(s)
    return (
        0.5 * h * jnp.dot(grad, grad)
        + h * jnp.trace(hess)
        + jnp.dot(grad_h, grad[d:])
    )


def mixed_objective(log_q: Callable, weight: Callable) -> Callable:
    """Build the sample version of J_MD in eq (2.9), constant dropped."""
    per_obs = jax.vmap(
        lambda y, s, theta: mixed_integrand(log_q, weight, y, s, theta),
        in_axes=(0, 0, None),
    )

    def jhat(theta, y, s):
        return jnp.mean(per_obs(y, s, theta))

    return jhat


def mixed_moment_fn(log_q: Callable, weight: Callable) -> Callable:
    """g_MD(y, s, theta), the theta-gradient of the eq (2.9) integrand.

    These are the estimating equations SME-BETEL uses for mixed-domain data,
    as described right after Proposition 1.
    """

    def g(y: jnp.ndarray, s: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(lambda th: mixed_integrand(log_q, weight, y, s, th))(theta)

    return g


def mixed_estimator(jmd: Callable, y: jnp.ndarray, s: jnp.ndarray, theta_init):
    """Minimize the sample version of J_MD in eq (2.9)."""
    return newton_minimize(lambda th: jmd(th, y, s), theta_init)
