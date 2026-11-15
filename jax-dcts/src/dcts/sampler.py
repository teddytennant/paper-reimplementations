"""Algorithm 1: Direct Conditional Transition Sampling (VE)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from dcts.operators import measurement_update
from dcts.statistic import flow_velocity, local_covariance, sufficient_statistic

Denoiser = Callable[[Array, Array], Array]


def inner_transition(
    eta: Array,
    x_t: Array,
    sigma_t: Array,
    sigma_next: Array,
    denoiser: Denoiser,
    y: Array,
    a: Array,
    sigma_y: Array,
    m_steps: int,
    sigma_src: Array,
    kappa: Array,
) -> Array:
    """One outer DCTS step: transport source noise eta to noise level sigma_next.

    Line 5 of Algorithm 1 samples r_0 = sigma_src eta. This helper takes
    the already-drawn unit Gaussian eta so tests can check the z-transport
    identity r_1 = z + sigma_next eta under an oracle denoiser.
    """
    r = sigma_src * eta
    h = 1.0 / m_steps
    dtype = x_t.dtype

    def body(j, r_s):
        s = jnp.asarray(j, dtype=dtype) / jnp.asarray(m_steps, dtype=dtype)
        x_tilde, sigma_tilde, v = sufficient_statistic(
            x_t, r_s, s, sigma_t, sigma_src, sigma_next
        )
        m_s = denoiser(x_tilde, sigma_tilde)
        c_s = local_covariance(v, kappa)
        m_y = measurement_update(m_s, c_s, y, a, sigma_y)
        u = flow_velocity(r_s, m_y, s, sigma_src, sigma_next)
        return r_s + h * u

    return jax.lax.fori_loop(0, m_steps, body, r)


def dcts(
    key: Array,
    denoiser: Denoiser,
    y: Array,
    a: Array,
    sigmas: Array,
    sigma_y: float = 0.05,
    m_steps: int = 2,
    sigma_src: float = 1.0,
    kappa: float = 1.0,
    x_init: Array | None = None,
) -> Array:
    """Run Algorithm 1.

    Parameters
    ----------
    key : PRNG key
    denoiser : D(x, sigma) approximating E[Z | X = x] under VE
    y : measurement, shape (m,)
    a : forward matrix, shape (m, d)
    sigmas : decreasing VE levels sigma_0 > ... > sigma_L, last typically 0
    sigma_y : observation noise std
    m_steps : inner Euler steps M
    sigma_src : source scale on the inner path
    kappa : covariance inflation on v_s
    x_init : optional x^{(0)}; default N(0, sigma_0^2 I)

    Returns
    -------
    x^{(L)}, the state after the last transition (clean if sigma_L = 0).
    """
    d = a.shape[1]
    dtype = jnp.asarray(y).dtype
    sigmas = jnp.asarray(sigmas, dtype=dtype)
    y = jnp.asarray(y, dtype=dtype)
    a = jnp.asarray(a, dtype=dtype)
    sigma_y_arr = jnp.asarray(sigma_y, dtype=dtype)
    sigma_src_arr = jnp.asarray(sigma_src, dtype=dtype)
    kappa_arr = jnp.asarray(kappa, dtype=dtype)

    key, key_x = jax.random.split(key)
    if x_init is None:
        x = sigmas[0] * jax.random.normal(key_x, (d,), dtype=dtype)
    else:
        x = jnp.asarray(x_init, dtype=dtype)

    n_outer = sigmas.shape[0] - 1

    def outer_body(k, carry):
        x_k, key_k = carry
        key_k, key_eta = jax.random.split(key_k)
        eta = jax.random.normal(key_eta, (d,), dtype=dtype)
        x_next = inner_transition(
            eta,
            x_k,
            sigmas[k],
            sigmas[k + 1],
            denoiser,
            y,
            a,
            sigma_y_arr,
            m_steps,
            sigma_src_arr,
            kappa_arr,
        )
        return x_next, key_k

    x_final, _ = jax.lax.fori_loop(0, n_outer, outer_body, (x, key))
    return x_final
