"""The SME-BETEL posterior and its Metropolis-Hastings sampler.

Equation (2.7) of arXiv:2609.01783 for the posterior, Algorithm S1 of
Appendix S4 for the sampler.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .etel import solve_tilt

__all__ = ["log_posterior_fn", "MHOutput", "metropolis_hastings"]


def log_posterior_fn(gmat: Callable, log_prior: Callable) -> Callable:
    """Build the unnormalized log SME-BETEL posterior of eq (2.7).

    log pi(theta | x) = log pi(theta) + log L(theta) up to a constant, where
    L is the ETEL function of eq (2.5). Returns -inf when the origin is not
    interior to the convex hull C(theta), which is the discard rule stated
    just after eq (2.7) and used in Algorithm S1.

    Args:
        gmat: maps (x, theta) to the (n, p) moment matrix.
        log_prior: maps theta to the log prior density, up to a constant.
    """

    def log_post(theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        sol = solve_tilt(gmat(x, theta))
        lp = log_prior(theta)
        total = lp + sol.log_lik
        ok = sol.feasible & jnp.isfinite(lp)
        return jnp.where(ok, total, -jnp.inf)

    return log_post


class MHOutput(NamedTuple):
    """Draws and diagnostics from Algorithm S1."""

    samples: jnp.ndarray
    accept_rate: jnp.ndarray
    final_scale: jnp.ndarray


def metropolis_hastings(
    log_target: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    key: jnp.ndarray,
    n_iter: int,
    burn_in: int,
    step_chol: jnp.ndarray | None = None,
    init_scale: float = 0.5,
    target_accept: float = 0.4,
    adapt: bool = True,
) -> MHOutput:
    """Algorithm S1, a random walk Metropolis-Hastings sampler for eq (2.7).

    The proposal is symmetric, so the p_prop ratio in Algorithm S1 cancels and
    r reduces to the ratio of pi(theta) L(theta). A proposal whose moment
    matrix does not contain the origin in the interior of its convex hull gets
    log_target = -inf and is rejected, which is the discard branch of the
    algorithm. During burn-in the scalar step scale is adapted toward
    target_accept, matching the roughly 0.4 acceptance rate the paper tunes to
    in Section 3.1; adaptation stops at burn-in so the retained draws come from
    a fixed kernel.
    """
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)
    p = theta0.shape[0]
    if step_chol is None:
        step_chol = jnp.eye(p)
    step_chol = jnp.asarray(step_chol, dtype=jnp.float64)

    def step(state, inputs):
        theta, logp, log_scale = state
        i, k = inputs
        k_prop, k_unif = jax.random.split(k)
        z = jax.random.normal(k_prop, (p,))
        prop = theta + jnp.exp(log_scale) * (step_chol @ z)
        logp_prop = log_target(prop)
        log_r = jnp.where(jnp.isfinite(logp_prop), logp_prop - logp, -jnp.inf)
        u = jnp.log(jax.random.uniform(k_unif))
        accept = u <= jnp.minimum(0.0, log_r)
        theta = jnp.where(accept, prop, theta)
        logp = jnp.where(accept, logp_prop, logp)
        if adapt:
            gamma = jnp.where(i < burn_in, 1.0 / (10.0 + i) ** 0.6, 0.0)
            log_scale = log_scale + gamma * (accept.astype(jnp.float64) - target_accept)
        return (theta, logp, log_scale), (theta, accept)

    keys = jax.random.split(key, n_iter)
    init = (theta0, log_target(theta0), jnp.log(jnp.asarray(init_scale)))
    (_, _, log_scale), (draws, accepts) = jax.lax.scan(
        step, init, (jnp.arange(n_iter), keys)
    )
    return MHOutput(
        samples=draws[burn_in:],
        accept_rate=jnp.mean(accepts[burn_in:].astype(jnp.float64)),
        final_scale=jnp.exp(log_scale),
    )
