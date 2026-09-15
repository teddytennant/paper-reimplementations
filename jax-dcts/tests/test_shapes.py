import jax
import jax.numpy as jnp

from dcts.denoisers import gaussian_ve_denoiser, mixture_ve_denoiser
from dcts.operators import (
    apply,
    apply_t,
    box_downsample_matrix,
    gaussian_matrix,
    inpainting_matrix,
)
from dcts.sampler import dcts
from dcts.schedule import edm_sigmas


def test_shapes_dtypes_finite():
    key = jax.random.PRNGKey(0)
    d = 12
    key_z, key_a, key_s = jax.random.split(key, 3)
    z = jax.random.normal(key_z, (d,))
    a = gaussian_matrix(key_a, m=8, d=d)
    y = a @ z
    sigmas = edm_sigmas(6, sigma_max=8.0, sigma_min=0.2)
    xhat = dcts(
        key_s,
        gaussian_ve_denoiser(1.0),
        y,
        a,
        sigmas,
        sigma_y=0.05,
        m_steps=2,
        sigma_src=1.0,
        kappa=1.0,
    )
    assert xhat.shape == (d,)
    assert xhat.dtype == jnp.float32
    assert jnp.isfinite(xhat).all()
    assert not jnp.any(jnp.isnan(xhat))


def test_operator_adjoints():
    key = jax.random.PRNGKey(1)
    d = 16
    x = jax.random.normal(key, (d,))
    a_g = gaussian_matrix(jax.random.PRNGKey(2), m=7, d=d)
    y_g = jax.random.normal(jax.random.PRNGKey(3), (7,))
    assert jnp.allclose(jnp.dot(apply(a_g, x), y_g), jnp.dot(x, apply_t(a_g, y_g)), atol=1e-5)

    keep = jnp.arange(d) % 2 == 0
    a_m = inpainting_matrix(keep)
    y_m = jax.random.normal(jax.random.PRNGKey(4), (a_m.shape[0],))
    assert jnp.allclose(jnp.dot(a_m @ x, y_m), jnp.dot(x, a_m.T @ y_m), atol=1e-5)

    a_s = box_downsample_matrix(4, 4, 2)
    y_s = jax.random.normal(jax.random.PRNGKey(5), (4,))
    assert a_s.shape == (4, 16)
    assert jnp.allclose(jnp.dot(a_s @ x, y_s), jnp.dot(x, a_s.T @ y_s), atol=1e-5)


def test_inpainting_and_sr_runs():
    key = jax.random.PRNGKey(6)
    d = 16
    z = jax.random.normal(key, (d,))
    keep = jnp.ones((d,), dtype=bool).at[::3].set(False)
    a = inpainting_matrix(keep)
    y = a @ z
    sigmas = edm_sigmas(8, sigma_max=10.0, sigma_min=0.15)
    xhat = dcts(
        jax.random.PRNGKey(7),
        gaussian_ve_denoiser(1.0),
        y,
        a,
        sigmas,
        sigma_y=0.03,
        m_steps=2,
        kappa=1.0,
    )
    assert xhat.shape == (d,)
    assert jnp.isfinite(xhat).all()

    a_sr = box_downsample_matrix(4, 4, 2)
    y_sr = a_sr @ z
    x_sr = dcts(
        jax.random.PRNGKey(8),
        mixture_ve_denoiser(jnp.stack([jnp.zeros((d,)), jnp.ones((d,))]), prior_var=0.5),
        y_sr,
        a_sr,
        sigmas,
        sigma_y=0.03,
        m_steps=2,
        kappa=1.0,
    )
    assert x_sr.shape == (d,)
    assert jnp.isfinite(x_sr).all()
