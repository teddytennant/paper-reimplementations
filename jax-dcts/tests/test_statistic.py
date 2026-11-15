import jax
import jax.numpy as jnp

from dcts.operators import gaussian_matrix, measurement_update
from dcts.statistic import (
    bar_sigma,
    local_covariance,
    precision_variance,
    sufficient_statistic,
)


def test_s_zero_recovers_current_state():
    key = jax.random.PRNGKey(0)
    x_t = jax.random.normal(key, (8,))
    r0 = 1.7 * jax.random.normal(jax.random.PRNGKey(1), (8,))
    sigma_t = jnp.array(3.0)
    x_tilde, sigma_tilde, v = sufficient_statistic(
        x_t, r0, s=jnp.array(0.0), sigma_t=sigma_t, sigma_src=jnp.array(1.7), sigma_next=jnp.array(0.4)
    )
    assert jnp.allclose(x_tilde, x_t, atol=1e-5)
    assert jnp.allclose(sigma_tilde, sigma_t, atol=1e-5)
    assert jnp.allclose(v, sigma_t**2, atol=1e-5)


def test_sufficient_statistic_is_ve_observation_of_z():
    """x_tilde = z + sigma_tilde * correlated_noise, variance v_s.

    With x_t = z + sigma_t eps and r_s = s z + bar_sigma eta, independent
    unit Gaussians, Eq. 7 says x_tilde - z has variance v_s = sigma_tilde^2.
    """
    key = jax.random.PRNGKey(2)
    d = 32
    n = 400
    z = jax.random.normal(key, (d,))
    s = jnp.array(0.4)
    sigma_t = jnp.array(2.0)
    sigma_src = jnp.array(1.0)
    sigma_next = jnp.array(0.5)
    sb = bar_sigma(s, sigma_src, sigma_next)
    v = precision_variance(s, sigma_t, sb)

    key_e, key_n = jax.random.split(jax.random.PRNGKey(3))
    eps = jax.random.normal(key_e, (n, d))
    eta = jax.random.normal(key_n, (n, d))
    x_t = z[None, :] + sigma_t * eps
    r_s = s * z[None, :] + sb * eta
    x_tilde, sigma_tilde, v_out = sufficient_statistic(
        x_t, r_s, s, sigma_t, sigma_src, sigma_next
    )
    err = x_tilde - z[None, :]
    emp_var = jnp.mean(err**2)
    assert jnp.allclose(v_out, v, atol=1e-5)
    assert jnp.allclose(sigma_tilde**2, v, atol=1e-5)
    assert jnp.allclose(emp_var, v, rtol=0.08, atol=0.05)


def test_covariance_scaling_and_operator_update():
    key = jax.random.PRNGKey(4)
    d = 6
    z = jax.random.normal(key, (d,))
    a = gaussian_matrix(jax.random.PRNGKey(5), m=6, d=d)
    y = a @ z
    s = jnp.array(0.5)
    sigma_t = jnp.array(1.5)
    sigma_src = jnp.array(1.0)
    sigma_next = jnp.array(0.2)
    sb = bar_sigma(s, sigma_src, sigma_next)
    v = precision_variance(s, sigma_t, sb)
    kappa = jnp.array(0.7)
    c = local_covariance(v, kappa)
    assert jnp.allclose(c, kappa * v)

    m = jnp.zeros((d,))
    m_y = measurement_update(m, c, y, a, sigma_y=jnp.array(1e-4))
    # large-c / tiny observation noise: update should nearly solve A m = y
    residual_before = jnp.linalg.norm(a @ m - y)
    residual_after = jnp.linalg.norm(a @ m_y - y)
    assert residual_after < 0.05 * residual_before
    assert jnp.isfinite(m_y).all()
