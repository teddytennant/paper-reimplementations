"""The synthetic validation setup of Section 4.1 and Appx. C.1, plus a small
random instance for the exact tests.

Appx. C.1 fixes everything: X = {0,...,240}, a Gaussian-shaped non-uniform
base p_0 centred at 68 with width 56, a latent score with a linear term and
three bumps at 60, 130 and 200, and a symmetric proposal that is a uniform
global move with probability 0.88 and a lazy nearest-neighbour move otherwise.
Since q is symmetric, r_0(k, k') = p_0(k') / p_0(k).
"""

import jax.numpy as jnp
import numpy as np

from .finite import DiscreteModel

N_STATES = 241
GAMMA = 0.88


def _gauss(k, a, tau):
    return np.exp(-0.5 * ((k - a) / tau) ** 2)


def base_log_p0(n_states=N_STATES):
    """log p_0 up to a constant, the Gaussian shape of Appx. C.1."""
    k = np.arange(n_states, dtype=np.float64)
    return -0.5 * ((k - 68.0) / 56.0) ** 2


def latent_score(n_states=N_STATES):
    """s(k) = 5k/240 + 0.9 g_{60,13} + 0.7 g_{130,21} + 1.0 g_{200,13}."""
    k = np.arange(n_states, dtype=np.float64)
    return (
        (5.0 / 240.0) * k
        + 0.9 * _gauss(k, 60.0, 13.0)
        + 0.7 * _gauss(k, 130.0, 21.0)
        + 1.0 * _gauss(k, 200.0, 13.0)
    )


def mixture_proposal(n_states=N_STATES, gamma=GAMMA):
    """q(k'|k) = gamma Unif + (1-gamma) L(k'|k), returned as log_q[j, i].

    L is the lazy nearest-neighbour move of Appx. C.1: L(k|k) = 1/2 and
    L(k+-1|k) = 1/4 in the interior, with mass that would leave the space kept
    at the current state, so L(0|0) = L(240|240) = 3/4. q is symmetric.
    """
    local = np.zeros((n_states, n_states))
    for i in range(n_states):
        local[i, i] += 0.5
        if i - 1 >= 0:
            local[i - 1, i] += 0.25
        else:
            local[i, i] += 0.25
        if i + 1 < n_states:
            local[i + 1, i] += 0.25
        else:
            local[i, i] += 0.25
    q = gamma / n_states + (1.0 - gamma) * local
    return np.log(q)


def synthetic_model(n_states=N_STATES, gamma=GAMMA):
    """The Appx. C.1 instance as a ``DiscreteModel``."""
    return DiscreteModel(
        log_p0=jnp.asarray(base_log_p0(n_states)),
        log_q=jnp.asarray(mixture_proposal(n_states, gamma)),
        scores=jnp.asarray(latent_score(n_states))[None, :],
    )


def random_model(seed, n_states=6, n_judges=1, symmetric=False, score_scale=1.0):
    """A small random instance for the exact tests.

    The proposal has every entry strictly positive, which gives Assumption 1's
    compatible support and target absolute continuity for free. With
    ``symmetric=False`` the proposal ratio does not cancel, so the Hastings
    part of r_0 is actually exercised.
    """
    rng = np.random.default_rng(seed)
    log_p0 = rng.normal(size=n_states) * 1.2
    q = rng.gamma(2.0, size=(n_states, n_states)) + 0.05
    if symmetric:
        q = q + q.T
    q = q / q.sum(axis=0, keepdims=True)  # column i is q(. | i)
    scores = rng.normal(size=(n_judges, n_states)) * score_scale
    return DiscreteModel(
        log_p0=jnp.asarray(log_p0),
        log_q=jnp.asarray(np.log(q)),
        scores=jnp.asarray(scores),
    )
