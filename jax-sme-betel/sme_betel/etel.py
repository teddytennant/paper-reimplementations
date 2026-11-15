"""Exponentially tilted empirical likelihood for a given moment matrix.

Section 2.2 of arXiv:2609.01783, equations (2.5) and (2.6), with the tilted
weights in Schennach (2005) form.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = ["ETELSolution", "solve_tilt", "etel_weights", "log_etel"]

_GRAD_TOL = 1e-10
_LAM_MAX = 1e8


class ETELSolution(NamedTuple):
    """Output of the ETEL dual problem.

    Attributes:
        lam: the tilting parameter lambda_hat(theta) of eq (2.6), original units.
        log_w: log of the tilted weights omega_i(theta), sums to one after exp.
        log_lik: sum_i log omega_i(theta), the log of L(theta) in eq (2.5) on the
            event that the origin lies inside the convex hull.
        feasible: whether the dual minimizer was attained. The infimum of the
            dual objective is attained if and only if the origin is interior to
            the convex hull C(theta) of eq (2.5), so this doubles as the convex
            hull check that Algorithm S1 needs.
        grad_norm: sup norm of the scaled dual gradient at the solution.
    """

    lam: jnp.ndarray
    log_w: jnp.ndarray
    log_lik: jnp.ndarray
    feasible: jnp.ndarray
    grad_norm: jnp.ndarray


def _dual(z: jnp.ndarray) -> jnp.ndarray:
    """log of (1/n) sum_i exp(lambda' g_i), given z = G lambda.

    Minimizing the log is the same problem as minimizing the sum in eq (2.6),
    and the log-sum-exp form does not overflow.
    """
    return logsumexp(z) - jnp.log(z.shape[0])


_ALPHAS = 0.25 ** jnp.arange(10)


def _backtrack(gs: jnp.ndarray, lam: jnp.ndarray, step: jnp.ndarray, f0: jnp.ndarray):
    """Largest step size in _ALPHAS that lowers the dual, or 0 if none does.

    A full Newton step is accepted almost every time once lambda is anywhere
    near the solution, so the common case costs one extra dual evaluation. The
    shrinking ladder is only evaluated when the full step fails, and it is
    evaluated in one matrix multiply rather than in a loop, which keeps the
    solve vmap friendly.
    """

    def dual_at(alpha):
        return _dual(gs @ (lam - alpha * step))

    def ladder(_):
        cands = lam[None, :] - _ALPHAS[:, None] * step[None, :]
        f = logsumexp(gs @ cands.T, axis=0) - jnp.log(gs.shape[0])
        ok = f < f0 - 1e-15 * jnp.abs(f0)
        return jnp.where(jnp.any(ok), _ALPHAS[jnp.argmax(ok)], 0.0)

    full_ok = dual_at(1.0) < f0 - 1e-15 * jnp.abs(f0)
    return jax.lax.cond(full_ok, lambda _: 1.0, ladder, operand=None)


def solve_tilt(g: jnp.ndarray, max_iter: int = 50) -> ETELSolution:
    """Solve for lambda_hat(theta) in eq (2.6) by damped Newton.

    The dual objective is convex in lambda with gradient sum_i w_i g_i and
    Hessian the w-weighted covariance of the g_i, so Newton converges in a
    handful of steps. Columns of g are rescaled to unit root mean square first,
    which makes the stopping tolerance independent of how g is parametrized.

    Args:
        g: moment matrix of shape (n, p), row i is g(x_i, theta).
    """
    g = jnp.asarray(g)
    _, p = g.shape
    scale = jnp.sqrt(jnp.mean(g**2, axis=0))
    scale = jnp.where(scale > 0, scale, 1.0)
    gs = g / scale

    def cond(state):
        lam, it, gnorm, stalled = state
        return (
            (it < max_iter)
            & (gnorm > _GRAD_TOL)
            & (jnp.max(jnp.abs(lam)) < _LAM_MAX)
            & (~stalled)
        )

    def body(state):
        lam, it, _, _ = state
        z = gs @ lam
        w = jax.nn.softmax(z)
        grad = gs.T @ w
        centered = gs - grad[None, :]
        hess = (centered * w[:, None]).T @ centered
        ridge = 1e-12 * jnp.maximum(1.0, jnp.trace(hess) / p)
        step = jnp.linalg.solve(hess + ridge * jnp.eye(p), grad)
        # Fall back on the gradient direction if the Newton step is not a descent
        # direction, which only happens when the Hessian is numerically singular.
        step = jnp.where(jnp.dot(step, grad) > 0, step, grad)
        alpha = _backtrack(gs, lam, step, _dual(z))
        stalled = alpha == 0.0
        new_lam = jnp.where(stalled, lam, lam - alpha * step)
        new_grad = gs.T @ jax.nn.softmax(gs @ new_lam)
        return (new_lam, it + 1, jnp.max(jnp.abs(new_grad)), stalled)

    lam0 = jnp.zeros(p)
    g0 = jnp.max(jnp.abs(jnp.mean(gs, axis=0)))
    lam_s, _, gnorm, _ = jax.lax.while_loop(cond, body, (lam0, 0, g0, False))

    z = gs @ lam_s
    log_w = z - logsumexp(z)
    feasible = (gnorm <= _GRAD_TOL) & (jnp.max(jnp.abs(lam_s)) < _LAM_MAX)
    return ETELSolution(
        lam=lam_s / scale,
        log_w=log_w,
        log_lik=jnp.sum(log_w),
        feasible=feasible,
        grad_norm=gnorm,
    )


def etel_weights(g: jnp.ndarray) -> jnp.ndarray:
    """The tilted probability weights omega(theta) of eq (2.6)."""
    return jnp.exp(solve_tilt(g).log_w)


def log_etel(g: jnp.ndarray) -> jnp.ndarray:
    """log L(theta) from eq (2.5), or -inf when the origin is outside C(theta)."""
    sol = solve_tilt(g)
    return jnp.where(sol.feasible, sol.log_lik, -jnp.inf)
