import jax
import jax.numpy as jnp

from adamx.opt import cosine_controller


def test_parallel_vectors_gamma_is_exp_lam():
    lam = 0.3
    g = jnp.array([2.0, -1.0, 0.5])
    gamma = cosine_controller(g, 3.0 * g, lam, 1e-8)
    assert jnp.allclose(gamma, jnp.exp(lam), atol=1e-6)


def test_opposite_vectors_gamma_is_exp_neg_lam():
    lam = 0.3
    g = jnp.array([2.0, -1.0, 0.5])
    gamma = cosine_controller(g, -g, lam, 1e-8)
    assert jnp.allclose(gamma, jnp.exp(-lam), atol=1e-6)


def test_orthogonal_2d_gamma_is_one():
    g = jnp.array([1.0, 0.0])
    g_prev = jnp.array([0.0, 1.0])
    gamma = cosine_controller(g, g_prev, 0.7, 1e-8)
    assert jnp.allclose(gamma, 1.0, atol=1e-6)


def test_zero_g_finite_no_nan():
    g = jnp.zeros((4,))
    g_prev = jnp.array([1.0, -2.0, 0.3, 4.0])
    gamma = cosine_controller(g, g_prev, 0.4, 1e-8)
    assert jnp.isfinite(gamma)
    assert not jnp.isnan(gamma)
    gamma0 = cosine_controller(g, jnp.zeros_like(g_prev), 0.4, 1e-8)
    assert jnp.isfinite(gamma0)
    assert not jnp.isnan(gamma0)


def test_gamma_always_in_exp_bounds():
    lam = 0.25
    lo, hi = jnp.exp(-lam), jnp.exp(lam)
    key = jax.random.PRNGKey(0)
    for _ in range(16):
        key, k1, k2 = jax.random.split(key, 3)
        g = jax.random.normal(k1, (8,))
        g_prev = jax.random.normal(k2, (8,))
        gamma = cosine_controller(g, g_prev, lam, 1e-8)
        assert gamma >= lo - 1e-6
        assert gamma <= hi + 1e-6
    nested = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([[-1.0], [0.5]])}
    gamma_n = cosine_controller(nested, nested, lam, 1e-8)
    assert jnp.allclose(gamma_n, hi, atol=1e-6)
