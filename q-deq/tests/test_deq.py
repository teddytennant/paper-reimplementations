"""Lock r = F(z, x) - z (eq 1)."""

import jax
import jax.numpy as jnp

from q_deq.deq import init_shared_params, residual, shared_layer


def test_residual_is_F_minus_z():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    hidden, input_dim = 16, 24
    params = init_shared_params(k1, hidden=hidden, input_dim=input_dim)
    z = jax.random.normal(k2, (hidden,))
    x = jax.random.normal(k3, (input_dim,))

    def F(zz, xx):
        return shared_layer(params, zz, xx)

    r = residual(F, z, x)
    assert r.shape == z.shape
    assert jnp.allclose(r, F(z, x) - z)
