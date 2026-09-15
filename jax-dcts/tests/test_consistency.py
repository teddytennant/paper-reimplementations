import jax
import jax.numpy as jnp

from dcts.denoisers import gaussian_ve_denoiser
from dcts.operators import gaussian_matrix
from dcts.sampler import dcts
from dcts.schedule import edm_sigmas


def test_measurement_consistency_vs_unconditioned():
    key = jax.random.PRNGKey(0)
    d = 16
    key_z, key_a, key_n, key_s, key_u = jax.random.split(key, 5)
    z = jax.random.normal(key_z, (d,))
    a = gaussian_matrix(key_a, m=16, d=d)
    sigma_y = 0.02
    y = a @ z + sigma_y * jax.random.normal(key_n, (16,))
    sigmas = edm_sigmas(18, sigma_max=20.0, sigma_min=0.1, rho=7.0)
    denoiser = gaussian_ve_denoiser(prior_var=1.0)

    x_cond = dcts(
        key_s,
        denoiser,
        y,
        a,
        sigmas,
        sigma_y=sigma_y,
        m_steps=3,
        sigma_src=1.0,
        kappa=1.0,
    )
    x_uncond = dcts(
        key_u,
        denoiser,
        y,
        a,
        sigmas,
        sigma_y=sigma_y,
        m_steps=3,
        sigma_src=1.0,
        kappa=0.0,
    )
    res_cond = jnp.linalg.norm(a @ x_cond - y)
    res_uncond = jnp.linalg.norm(a @ x_uncond - y)
    assert jnp.isfinite(x_cond).all()
    assert res_cond < res_uncond
    assert res_cond < 0.5 * res_uncond


def test_reconstruction_beats_noise():
    key = jax.random.PRNGKey(1)
    d = 16
    key_z, key_a, key_n, key_s, key_noise = jax.random.split(key, 5)
    z = jax.random.normal(key_z, (d,))
    a = gaussian_matrix(key_a, m=16, d=d)
    sigma_y = 0.02
    y = a @ z + sigma_y * jax.random.normal(key_n, (16,))
    sigmas = edm_sigmas(18, sigma_max=20.0, sigma_min=0.1, rho=7.0)
    xhat = dcts(
        key_s,
        gaussian_ve_denoiser(1.0),
        y,
        a,
        sigmas,
        sigma_y=sigma_y,
        m_steps=3,
        sigma_src=1.0,
        kappa=1.0,
    )
    noise = jax.random.normal(key_noise, (d,))
    mse_hat = jnp.mean((xhat - z) ** 2)
    mse_noise = jnp.mean((noise - z) ** 2)
    assert jnp.isfinite(xhat).all()
    assert mse_hat < mse_noise
    assert mse_hat < 0.25
