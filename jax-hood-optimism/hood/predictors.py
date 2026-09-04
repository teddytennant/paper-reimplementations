"""The (N+1)-th order discounted predictor, Section 3.2 and Appendix B.1.

Equation (22) defines the prediction from the last N+1 payoff vectors and the
last N+1 predictions,

    m_{i,t} = sum_{k=1}^{N+1} (-1)^{k+1} C(N+1, k)
              [ (1 - delta^k) g_{i,t-k} + delta^k m_{i,t-k} ],

with the discount delta = N / (N + 1) of eq. (23), and the convention
g_{i,k} = m_{i,k} = 0 for -N <= k <= 0 from line 1 of Algorithm 1. The
prediction error is e_{i,t} = g_{i,t} - m_{i,t}, eq. (24).

Lemma B.1 says the point of all this: with sigma the geometric discounted sum
(sigma w)_t = sum_{k>=0} delta^k w_{t-k} and Delta the backward difference,

    e_i = sigma^{N+1} Delta^{N+1} g_i.                              eq. (B.5)

So the error carries N+1 time differences of the payoff sequence, and the
discounting keeps the N+1 accumulations from blowing up with N.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


def discount_factor(num_players: int) -> float:
    """delta = N / (N + 1), eq. (23)."""
    return num_players / (num_players + 1.0)


def predictor_coefficients(num_players: int) -> np.ndarray:
    """The signed binomials (-1)^{k+1} C(N+1, k) of eq. (22), for k = 1..N+1."""
    order = num_players + 1
    return np.array(
        [(-1.0) ** (k + 1) * math.comb(order, k) for k in range(1, order + 1)],
        dtype=np.float64,
    )


def hood_predictor(
    g_buffer: jnp.ndarray, m_buffer: jnp.ndarray, num_players: int
) -> jnp.ndarray:
    """One evaluation of eq. (22).

    Args:
        g_buffer: array of shape (N+1, ...) holding g_{t-1}, ..., g_{t-N-1}.
        m_buffer: array of shape (N+1, ...) holding m_{t-1}, ..., m_{t-N-1}.
        num_players: N.

    Returns:
        The prediction m_t, with the shape of a single buffer entry.
    """
    delta = discount_factor(num_players)
    k = np.arange(1, num_players + 2, dtype=np.float64)
    delta_k = jnp.asarray(delta**k)
    coef = jnp.asarray(predictor_coefficients(num_players))
    weights_g = coef * (1.0 - delta_k)
    weights_m = coef * delta_k
    trailing = (1,) * (g_buffer.ndim - 1)
    return jnp.sum(
        weights_g.reshape((-1,) + trailing) * g_buffer
        + weights_m.reshape((-1,) + trailing) * m_buffer,
        axis=0,
    )


def push(buffer: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Roll a lag buffer forward one round, newest entry first."""
    return jnp.concatenate([value[None], buffer[:-1]], axis=0)


def predictor_sequence(g: jnp.ndarray, num_players: int) -> jnp.ndarray:
    """Run eq. (22) over a whole payoff sequence, zero prehistory.

    Args:
        g: array of shape (T, ...) holding g_1, ..., g_T.
        num_players: N.

    Returns:
        Array of shape (T, ...) holding m_1, ..., m_T.
    """
    order = num_players + 1
    zeros = jnp.zeros((order,) + g.shape[1:], dtype=g.dtype)

    def step(carry, g_t):
        g_buf, m_buf = carry
        m_t = hood_predictor(g_buf, m_buf, num_players)
        return (push(g_buf, g_t), push(m_buf, m_t)), m_t

    _, m = jax.lax.scan(step, (zeros, zeros), g)
    return m


def backward_difference(w: jnp.ndarray) -> jnp.ndarray:
    """(Delta w)_t = w_t - w_{t-1} with zero prehistory, eq. (B.1)."""
    return w - jnp.concatenate([jnp.zeros_like(w[:1]), w[:-1]], axis=0)


def discounted_sum(w: jnp.ndarray, delta: float) -> jnp.ndarray:
    """(sigma w)_t = sum_{k>=0} delta^k w_{t-k} with zero prehistory, eq. (B.1)."""

    def step(carry, w_t):
        carry = w_t + delta * carry
        return carry, carry

    _, out = jax.lax.scan(step, jnp.zeros_like(w[0]), w)
    return out


def discounted_difference(w: jnp.ndarray, num_players: int) -> jnp.ndarray:
    """sigma^{N+1} Delta^{N+1} w, the right-hand side of eq. (B.5)."""
    delta = discount_factor(num_players)
    out = w
    for _ in range(num_players + 1):
        out = backward_difference(out)
    for _ in range(num_players + 1):
        out = discounted_sum(out, delta)
    return out


def first_order_predictor(
    g_buffer: jnp.ndarray, m_buffer: jnp.ndarray, num_players: int
) -> jnp.ndarray:
    """The one-step-ahead predictor m_t = g_{t-1} of ordinary OptFTRL.

    Same signature as :func:`hood_predictor` so the two drop into the same
    loop. This is the predictor behind optimistic multiplicative weights.
    """
    del m_buffer, num_players
    return g_buffer[0]
