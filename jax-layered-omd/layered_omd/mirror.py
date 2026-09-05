"""The smoothed-entropy mirror map and its proximal step.

Section 3.1 of arXiv:2609.00504 regularizes with the smoothed negative entropy

    Psi_i(x) = sum_a (x(a) + lambda_i) log(x(a) + lambda_i),

whose Bregman divergence is

    D_i(x, y) = sum_a (x(a) + lambda_i) log((x(a) + lambda_i)/(y(a) + lambda_i)).

Both mirror steps of Algorithm 1 (lines 6 and 11) are the same map

    P_y(g) = argmin_{x in X} { eta <g, x> + D_i(x, y) },

with ``X = Delta(A_i)`` in the full-feedback algorithm and the floored simplex
``Delta_i^zeta = {x in Delta(A_i) : x(a) >= zeta/|A_i|}`` in Algorithm 2.

The step has a water-filling solution. Writing ``w_a = (y_a + lambda) e^{-eta g_a}``
and ``c`` for the floor, the KKT conditions of the (strictly convex) program give

    x_a = max(theta w_a - lambda, c),   theta > 0 chosen so that sum_a x_a = 1.

The left-hand side is increasing in ``theta`` with breakpoints at
``theta = (c + lambda)/w_a``, so sorting ``w`` descending and testing each
prefix as the unclipped set finds ``theta`` exactly. At ``lambda = 0`` and
``c = 0`` no coordinate ever clips and the step collapses to the usual
exponential-weights update ``x = y e^{-eta g} / Z``.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "bregman_divergence",
    "prox_step",
    "step_sizes",
    "prox_constant",
    "bregman_diameter",
    "regret_constant",
]


def bregman_divergence(x, y, lam):
    """``D_i(x, y)`` for the smoothed negative entropy (equation 2)."""
    xs, ys = x + lam, y + lam
    return jnp.sum(xs * (jnp.log(xs) - jnp.log(ys)), axis=-1)


def prox_constant(n_actions, lam):
    """``rho_i = 1 + |A_i| lambda_i``, the strong-convexity constant of Lemma 4."""
    return 1.0 + n_actions * lam


def bregman_diameter(n_actions, lam):
    """``B_i = rho_i log((1 + lambda_i)/lambda_i)``, the diameter bound of Lemma 4."""
    return prox_constant(n_actions, lam) * jnp.log((1.0 + lam) / lam)


def prox_step(grad, base, eta, lam, floor=0.0):
    """One smoothed-entropy mirror step over the (possibly floored) simplex.

    Args:
        grad: ``g``, the linear part, shape ``(..., n)``.
        base: ``y``, the point the divergence is measured from, shape ``(..., n)``.
        eta: step size, broadcastable against the leading axes.
        lam: smoothing parameter ``lambda_i > 0`` (``0`` gives plain entropy).
        floor: per-coordinate lower bound ``c = zeta/|A_i|``; must satisfy
            ``n * floor < 1``.

    Returns:
        ``argmin_{x >= floor, sum x = 1} { eta <g, x> + D(x, base) }``.
    """
    grad = jnp.asarray(grad)
    base = jnp.asarray(base)
    n = grad.shape[-1]
    eta = jnp.asarray(eta, dtype=grad.dtype)
    while eta.ndim < grad.ndim:
        eta = eta[..., None]

    log_w = jnp.log(base + lam) - eta * grad
    log_w = log_w - jnp.max(log_w, axis=-1, keepdims=True)
    w = jnp.exp(log_w)

    # Candidate solutions: the k largest coordinates of w stay unclipped.
    w_sorted = jnp.sort(w, axis=-1)[..., ::-1]
    prefix = jnp.cumsum(w_sorted, axis=-1)
    k = jnp.arange(1, n + 1, dtype=grad.dtype)
    theta = (1.0 + k * lam - (n - k) * floor) / prefix

    knee = floor + lam
    lower_ok = theta * w_sorted >= knee
    nxt = jnp.concatenate(
        [w_sorted[..., 1:], jnp.zeros(w_sorted.shape[:-1] + (1,), grad.dtype)], axis=-1
    )
    upper_ok = theta * nxt <= knee
    feasible = lower_ok & upper_ok
    # argmax picks the first feasible k; ties give the same theta.
    idx = jnp.argmax(feasible, axis=-1)
    theta = jnp.take_along_axis(theta, idx[..., None], axis=-1)

    return jnp.maximum(theta * w - lam, floor)


def step_sizes(horizon, n_rounds, eta0):
    """The increasing layer-wise schedule of Algorithm 1, line 2.

    ``alpha_h = (3(H - h) + 1)/(3H + 1)`` and ``eta_h = eta_0 T^{-alpha_h}`` for
    ``h = 1..H``. ``alpha`` decreases in ``h``, so deep layers move fastest and
    early layers stay nearly frozen; that is what keeps the drift of the hidden
    history-occupancy weights of Lemma 9 under control. Returns an array indexed
    by ``h - 1``.
    """
    h = jnp.arange(1, horizon + 1, dtype=jnp.result_type(float))
    alpha = (3.0 * (horizon - h) + 1.0) / (3.0 * horizon + 1.0)
    return eta0 * jnp.asarray(n_rounds, dtype=alpha.dtype) ** (-alpha)


def regret_constant(horizon, n_players, n_actions_max, eta0):
    """``C_*^SE`` of Theorem 1, the constant in ``Reg_T <= C_*^SE T^{(3H-2)/(3H+1)}``."""
    log_a = jnp.log(n_actions_max + 1.0)
    h = float(horizon)
    return (
        2.0 * h * log_a / eta0
        + 12.0 * n_players * h ** 3 * log_a
        + eta0 * h ** 3
        + 144.0 * eta0 ** 3 * n_players ** 2 * h ** 7
    )
