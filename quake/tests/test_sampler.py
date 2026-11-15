"""Plug-in corrector loop (Algorithm 1)."""

import jax.numpy as jnp
import numpy as np

from quake.sampler import correct_loop
from quake.transition import shift_register


def test_plugin_identity_observations_stay_close():
    """When f̂ = f (γ=1, ξ=0, R tiny) the posterior mean of the current slot stays close to f."""
    T = 12
    window = 2
    f = jnp.sin(jnp.arange(T, dtype=jnp.float32) / 2.0)
    A, b, G = shift_register(jnp.array([0.8, 0.2]), 0.0, window)
    init_mean = jnp.array([f[0], f[0]])
    init_cov = jnp.eye(window) * 0.25
    f_corr, _state = correct_loop(
        f,
        init_mean,
        init_cov,
        A,
        b,
        G,
        mu=0.0,
        Q=0.05,
        gamma=1.0,
        xi=0.0,
        R=1e-8,
    )
    np.testing.assert_allclose(np.asarray(f_corr), np.asarray(f), atol=1e-4)
