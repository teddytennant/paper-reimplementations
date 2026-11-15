import jax
import jax.numpy as jnp
import numpy as np

from pointlam.grid import neighbor_mask
from pointlam.lha import (
    apply_lha,
    init_lha,
    layer_norm,
    local_hadamard,
    residual_hadamard,
)


def test_lha_shape_matches_input():
    key = jax.random.PRNGKey(0)
    n, dim = 12, 8
    points = jax.random.uniform(key, (n, 3))
    features = jax.random.normal(jax.random.fold_in(key, 1), (n, dim))
    params = init_lha(jax.random.fold_in(key, 2), dim)
    out = apply_lha(params, points, features, cell_size=0.4)
    assert out.shape == features.shape
    assert jnp.isfinite(out).all()


def test_lha_wv_zero_is_layernorm_of_input():
    key = jax.random.PRNGKey(3)
    n, dim = 10, 6
    points = jax.random.uniform(key, (n, 3))
    features = jax.random.normal(jax.random.fold_in(key, 1), (n, dim))
    params = init_lha(jax.random.fold_in(key, 2), dim)
    params = params._replace(W_v=jnp.zeros_like(params.W_v))
    mask = neighbor_mask(points, 0.5)
    f_prime = local_hadamard(points, features, mask, params.W_v, params.W_g)
    np.testing.assert_allclose(np.array(f_prime), 0.0, atol=1e-6)
    out = apply_lha(params, points, features, cell_size=0.5, mask=mask)
    expected = layer_norm(features, params.ln_gamma, params.ln_beta)
    np.testing.assert_allclose(np.array(out), np.array(expected), atol=1e-5, rtol=1e-5)


def test_eq5_isolation():
    key = jax.random.PRNGKey(4)
    n, dim = 7, 5
    k1, k2, k3, k4 = jax.random.split(key, 4)
    f_in = jax.random.normal(k1, (n, dim))
    f_prime = jax.random.normal(k2, (n, dim))
    W1 = jax.random.normal(k3, (dim, dim))
    W2 = jax.random.normal(k4, (dim, dim))
    gamma = jnp.linspace(0.5, 1.5, dim)
    beta = jnp.linspace(-0.2, 0.3, dim)
    out = residual_hadamard(f_in, f_prime, W1, W2, gamma, beta, eps=1e-5)
    inner = f_in + (f_prime @ W1) * jax.nn.sigmoid(f_in @ W2)
    expected = layer_norm(inner, gamma, beta, 1e-5)
    np.testing.assert_allclose(np.array(out), np.array(expected), atol=1e-6, rtol=1e-6)
