"""RealNVP invertibility, logdet, importance weights (8), and RESS (9)."""

import jax
import jax.numpy as jnp
import numpy as np

from genaimmd.bg import importance_weights, ress
from genaimmd.flow import forward, init_realnvp, inverse


def test_realnvp_inverse_forward_and_logdet():
    params = init_realnvp(jax.random.PRNGKey(4))
    z = jax.random.normal(jax.random.PRNGKey(5), (16, 2))
    c = jnp.full((16,), 0.5)
    x, logdet_fwd = forward(params, z, c)
    z_rec, logdet_inv = inverse(params, x, c)
    np.testing.assert_allclose(np.asarray(z_rec), np.asarray(z), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(logdet_fwd + logdet_inv), np.zeros(16), atol=1e-4
    )


def test_untrained_ress_and_positive_weights():
    params = init_realnvp(jax.random.PRNGKey(6))
    z = jax.random.normal(jax.random.PRNGKey(7), (32, 2))
    c = jnp.full((32,), 0.5)
    w = importance_weights(params, z, c)
    assert jnp.all(w > 0.0)
    r = float(ress(w))
    assert 0.0 < r <= 1.0 + 1e-6
