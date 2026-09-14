import jax
import jax.numpy as jnp

from klxx.flow import forward, init_coupling, inverse


def test_coupling_inverse_recovers_and_logdet_matches_autodiff():
    key = jax.random.PRNGKey(1)
    key, k_init, k_pert, k_y = jax.random.split(key, 4)
    params = init_coupling(k_init, dim=2, n_layers=4, hidden=8)
    # Perturb away from the identity so the jacobian is nontrivial.
    params = jax.tree.map(lambda p: p + 0.2 * jax.random.normal(k_pert, p.shape), params)
    y = jax.random.normal(k_y, (2,))
    x, logdet = forward(params, y)
    y_rec, logdet_inv = inverse(params, x)
    assert jnp.allclose(y_rec, y, atol=1e-5)
    assert jnp.allclose(logdet + logdet_inv, 0.0, atol=1e-5)

    def g_only(yy):
        xx, _ = forward(params, yy)
        return xx

    jac = jax.jacfwd(g_only)(y)
    logdet_ad = jnp.linalg.slogdet(jac)[1]
    assert jnp.allclose(logdet, logdet_ad, atol=1e-4)


def test_batched_inverse_roundtrip():
    key = jax.random.PRNGKey(2)
    key, k_init, k_x = jax.random.split(key, 3)
    params = init_coupling(k_init, dim=2, n_layers=6, hidden=16)
    x = jax.random.normal(k_x, (12, 2))
    y, _ = inverse(params, x)
    x_rec, _ = forward(params, y)
    assert jnp.allclose(x_rec, x, atol=1e-5)
