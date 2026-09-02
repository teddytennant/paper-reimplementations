"""Sec. II-B, the prior-art score function of the noised finite dataset.

The forward process (1) turns the dataset into a Gaussian mixture, eq (2). Its
score has the closed form (4)-(5), and the probability-flow ODE (3) runs that
score backward in time to generate a sample. The last function here is the piece
the paper points at in Sec. II-C: the posterior mean implied by (4) is exactly
eq (11), with sigma^2 = (1 - alpha_t) / alpha_t.

Shapes match `closed_form`: `data` is (N, d) and a query `y` is (..., d).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .closed_form import optimal_denoiser

__all__ = [
    "sigma_from_alpha",
    "mixture_log_density",
    "mixture_weights",
    "score",
    "posterior_mean",
    "posterior_mean_via_closed_form",
    "probability_flow_rhs",
    "reverse_ode",
]


def sigma_from_alpha(alpha_t):
    """The sigma of eq (11) that matches the noise level alpha_t of eq (1).

    Eq (1) has y = sqrt(alpha) x0 + sqrt(1 - alpha) n. Dividing by sqrt(alpha)
    puts it in the y = s + n form of eq (6) with variance (1 - alpha) / alpha.
    """
    alpha_t = jnp.asarray(alpha_t)
    return jnp.sqrt((1.0 - alpha_t) / alpha_t)


def _log_kernel(y, data, alpha_t):
    """log N(y | sqrt(alpha_t) s, (1 - alpha_t) I) for every s. Returns (..., N)."""
    y = jnp.asarray(y)
    data = jnp.asarray(data)
    alpha_t = jnp.asarray(alpha_t)
    var = 1.0 - alpha_t
    diff = y[..., None, :] - jnp.sqrt(alpha_t) * data
    d2 = jnp.sum(diff * diff, axis=-1)
    dim = data.shape[-1]
    return -d2 / (2.0 * var) - 0.5 * dim * jnp.log(2.0 * jnp.pi * var)


def mixture_log_density(y, data, alpha_t):
    """log p_t(y) for the Gaussian mixture of eq (2). Returns (...,)."""
    n = jnp.asarray(data).shape[0]
    return logsumexp(_log_kernel(y, data, alpha_t), axis=-1) - jnp.log(n)


def mixture_weights(y, data, alpha_t):
    """W_t(y, s) of eq (5), the responsibility of each mixture component. (..., N)."""
    log_k = _log_kernel(y, data, alpha_t)
    return jnp.exp(log_k - logsumexp(log_k, axis=-1, keepdims=True))


def score(y, data, alpha_t):
    """Eq (4): s_t(y) = 1/(1 - alpha_t) sum_s (sqrt(alpha_t) s - y) W_t(y, s).

    This is grad_y log p_t(y) for the mixture (2), checked against finite
    differences in the tests.
    """
    y = jnp.asarray(y)
    data = jnp.asarray(data)
    alpha_t = jnp.asarray(alpha_t)
    w = mixture_weights(y, data, alpha_t)
    return (jnp.sqrt(alpha_t) * (w @ data) - y) / (1.0 - alpha_t)


def posterior_mean(y, data, alpha_t):
    """E[x0 | y] read off eq (4), which is eq (11) at sigma = sigma_from_alpha.

    Tweedie's identity for eq (1) says
        grad log p_t(y) = (sqrt(alpha) E[x0|y] - y) / (1 - alpha),
    and comparing with eq (4) gives E[x0|y] = sum_s s W_t(y, s). Rescaling the
    query by 1/sqrt(alpha) turns W_t into the p_i of eq (11). That is the link
    the paper mentions after eq (15): (11) can be deduced from (4) and (5).
    """
    y = jnp.asarray(y)
    alpha_t = jnp.asarray(alpha_t)
    return (y + (1.0 - alpha_t) * score(y, data, alpha_t)) / jnp.sqrt(alpha_t)


def posterior_mean_via_closed_form(y, data, alpha_t):
    """The same quantity through eq (11), for the equivalence test."""
    alpha_t = jnp.asarray(alpha_t)
    return optimal_denoiser(
        jnp.asarray(y) / jnp.sqrt(alpha_t), data, sigma_from_alpha(alpha_t)
    )


def probability_flow_rhs(y, data, alpha_t, gamma_t=1.0):
    """Eq (3): -y_dot = gamma_t (y + s_t(y)), returned as y_dot.

    Forward in time this flow blows up; generation runs it backward, which is
    what `reverse_ode` does. With a single dataset point the flow has a fixed
    point at s / sqrt(alpha_t), asserted in the tests.
    """
    y = jnp.asarray(y)
    return -jnp.asarray(gamma_t) * (y + score(y, data, alpha_t))


def reverse_ode(y, data, alphas, gammas=None, dt=1.0):
    """Integrate eq (3) backward in time with explicit Euler.

    `alphas` is the schedule visited in reverse order, from the noisiest level to
    the cleanest. Returns the whole trajectory, shape (len(alphas) + 1, ..., d).
    """
    alphas = jnp.asarray(alphas)
    if gammas is None:
        gammas = jnp.ones_like(alphas)
    gammas = jnp.broadcast_to(jnp.asarray(gammas), alphas.shape)
    traj = [jnp.asarray(y)]
    for alpha_t, gamma_t in zip(alphas, gammas):
        y = y - dt * probability_flow_rhs(y, data, alpha_t, gamma_t)
        traj.append(y)
    return jnp.stack(traj)
