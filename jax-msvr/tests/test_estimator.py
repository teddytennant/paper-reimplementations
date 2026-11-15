"""Tracking behaviour of MSVR vs SOX."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_msvr.estimator import correction_gamma, msvr_update, sox_update
from jax_msvr.metrics import tracking_error


def test_msvr_tracks_better_than_sox_on_slow_w():
    m, k, d, B1 = 8, 1, 4, 1
    beta = 0.3
    sigma = 0.15
    T = 400
    gamma = float(correction_gamma(m, B1, beta))
    key = jax.random.PRNGKey(0)
    key, kA, kw = jax.random.split(key, 3)
    A = 0.4 * jax.random.normal(kA, (m, k, d))
    w0 = jax.random.normal(kw, (d,))
    direction = jnp.ones(d) / jnp.sqrt(d)
    drift = 0.002

    def run(use_msvr):
        u = jnp.zeros((m, k))
        w = w0
        w_prev = w0
        k_loop = jax.random.PRNGKey(123)
        for _ in range(T):
            k_loop, k_idx, k_noise = jax.random.split(k_loop, 3)
            idx = jax.random.choice(k_idx, m, shape=(B1,), replace=False)
            noise = sigma * jax.random.normal(k_noise, (B1, k))
            g_now = jnp.einsum("bkd,d->bk", A[idx], w) + noise
            g_prev = jnp.einsum("bkd,d->bk", A[idx], w_prev) + noise
            if use_msvr:
                u = msvr_update(u, g_now, g_prev, idx, beta, gamma)
            else:
                u = sox_update(u, g_now, idx, beta)
            w_prev = w
            w = w + drift * direction
        g_exact = jnp.einsum("mkd,d->mk", A, w_prev)
        return float(tracking_error(u, g_exact))

    err_msvr = run(True)
    err_sox = run(False)
    err_init = float(tracking_error(jnp.zeros((m, k)), jnp.einsum("mkd,d->mk", A, w0)))
    assert err_msvr < err_sox or err_msvr < err_init
    assert err_msvr < err_sox


def test_beta_greater_than_half_does_not_crash():
    m, B1, beta = 10, 2, 0.7
    gamma = float(correction_gamma(m, B1, beta))
    assert np.isfinite(gamma)
    u = jnp.zeros((m, 2))
    g_now = jnp.ones((B1, 2))
    g_prev = jnp.zeros((B1, 2))
    idx = jnp.array([0, 3])
    u_new = msvr_update(u, g_now, g_prev, idx, beta, gamma)
    assert np.isfinite(np.asarray(u_new)).all()
