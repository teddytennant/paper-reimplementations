import jax
import jax.numpy as jnp

from dcts.denoisers import oracle_denoiser
from dcts.operators import gaussian_matrix
from dcts.sampler import inner_transition


def test_oracle_euler_hits_refresh_kernel():
    """Paper equivalent of x_{t'} = alpha_{t'} z: under VE, alpha ≡ 1, so
    r_1 = z + sigma_next eta when the inner mean is the true z.

    The inner path is linear in s, so Euler with step 1/M is exact.
    """
    key = jax.random.PRNGKey(0)
    d = 16
    z = jax.random.normal(key, (d,))
    eta = jax.random.normal(jax.random.PRNGKey(1), (d,))
    a = jnp.zeros((1, d))
    y = jnp.zeros((1,))
    sigma_t = jnp.array(2.0)
    sigma_next = jnp.array(0.35)
    x_t = z + sigma_t * jax.random.normal(jax.random.PRNGKey(2), (d,))
    r1 = inner_transition(
        eta,
        x_t,
        sigma_t,
        sigma_next,
        oracle_denoiser(z),
        y,
        a,
        sigma_y=jnp.array(1.0),
        m_steps=4,
        sigma_src=jnp.array(1.0),
        kappa=jnp.array(0.0),
    )
    target = z + sigma_next * eta
    assert jnp.allclose(r1, target, atol=1e-5)


def test_point_refresh_m1():
    """M = 1 is point refresh: r_1 = m^y + sigma_next eta with m^y = z here."""
    d = 8
    z = jnp.arange(d, dtype=jnp.float32) * 0.1
    eta = jnp.ones((d,))
    sigma_next = jnp.array(0.5)
    r1 = inner_transition(
        eta,
        z + 3.0 * eta,
        jnp.array(3.0),
        sigma_next,
        oracle_denoiser(z),
        y=jnp.zeros((1,)),
        a=jnp.zeros((1, d)),
        sigma_y=jnp.array(1.0),
        m_steps=1,
        sigma_src=jnp.array(1.2),
        kappa=jnp.array(0.0),
    )
    assert jnp.allclose(r1, z + sigma_next * eta, atol=1e-5)


def test_transport_with_consistent_measurements():
    key = jax.random.PRNGKey(3)
    d = 10
    z = jax.random.normal(key, (d,))
    a = gaussian_matrix(jax.random.PRNGKey(4), m=10, d=d)
    y = a @ z
    eta = jax.random.normal(jax.random.PRNGKey(5), (d,))
    sigma_next = jnp.array(0.2)
    r1 = inner_transition(
        eta,
        z + 1.5 * jax.random.normal(jax.random.PRNGKey(6), (d,)),
        jnp.array(1.5),
        sigma_next,
        oracle_denoiser(z),
        y,
        a,
        sigma_y=jnp.array(1e-5),
        m_steps=5,
        sigma_src=jnp.array(1.0),
        kappa=jnp.array(1.0),
    )
    assert jnp.allclose(r1, z + sigma_next * eta, atol=2e-3)
