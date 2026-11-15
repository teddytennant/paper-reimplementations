import jax.numpy as jnp

from dcts.schedule import edm_sigmas, ve_alpha, vp_alpha_sigma, vp_bar_alpha, vp_discrete


def test_vp_identity_alpha_sq_plus_sigma_sq():
    t = jnp.linspace(0.0, 1.0, 21)
    alpha, sigma = vp_alpha_sigma(t)
    assert jnp.allclose(alpha**2 + sigma**2, 1.0, atol=1e-6)


def test_vp_monotonic_and_endpoints():
    t = jnp.linspace(0.0, 1.0, 17)
    alpha, sigma = vp_alpha_sigma(t)
    assert jnp.isclose(alpha[0], 1.0, atol=1e-6)
    assert jnp.isclose(sigma[0], 0.0, atol=1e-6)
    assert jnp.isclose(alpha[-1], 0.0, atol=1e-6)
    assert jnp.isclose(sigma[-1], 1.0, atol=1e-6)
    assert jnp.all(jnp.diff(alpha) <= 1e-6)
    assert jnp.all(jnp.diff(sigma) >= -1e-6)


def test_vp_bar_alpha_is_alpha_squared():
    t = jnp.array([0.0, 0.3, 0.8, 1.0])
    alpha, _sigma = vp_alpha_sigma(t)
    assert jnp.allclose(vp_bar_alpha(t), alpha**2, atol=1e-6)


def test_vp_discrete_decreasing_noise():
    alpha, sigma = vp_discrete(8)
    assert alpha.shape == (9,)
    assert sigma.shape == (9,)
    assert jnp.all(jnp.diff(sigma) <= 1e-6)
    assert jnp.allclose(alpha**2 + sigma**2, 1.0, atol=1e-6)


def test_ve_alpha_is_one():
    sigmas = edm_sigmas(10, sigma_max=50.0, sigma_min=0.1)
    assert jnp.allclose(ve_alpha(sigmas), 1.0)


def test_edm_monotonic_endpoints():
    sigmas = edm_sigmas(12, sigma_max=100.0, sigma_min=0.1, rho=7.0)
    assert sigmas.shape == (13,)
    assert jnp.isclose(sigmas[0], 100.0, rtol=1e-5)
    assert jnp.isclose(sigmas[-2], 0.1, rtol=1e-5)
    assert jnp.isclose(sigmas[-1], 0.0, atol=1e-8)
    assert jnp.all(jnp.diff(sigmas) <= 1e-6)
