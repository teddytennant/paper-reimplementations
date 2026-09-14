import jax
import jax.numpy as jnp
import numpy as np

from klxx.flow import forward, init_coupling, source_potential, two_mode_potential
from klxx.loss import klxx_loss, log_ratio_variation, log_ratio_z, z_from_flow


def test_variation_zero_on_constant_z():
    z = jnp.full((8,), 3.14)
    assert jnp.allclose(log_ratio_variation(z), 0.0, atol=1e-6)


def test_variation_matches_brute_force_n5():
    z = jnp.array([0.4, -1.2, 2.0, 0.0, 0.7])
    n = z.shape[0]
    brute = 0.0
    zz = np.asarray(z)
    for i in range(n):
        for j in range(n):
            if i != j:
                brute += abs(float(zz[i] - zz[j]))
    brute /= n * (n - 1)
    got = float(log_ratio_variation(z))
    assert abs(got - brute) < 1e-6


def test_variation_invariant_to_additive_shift():
    z = jnp.array([1.0, -0.5, 2.2, 0.3, -1.7, 4.0])
    c = 12.5
    assert jnp.allclose(log_ratio_variation(z), log_ratio_variation(z + c), atol=1e-6)


def test_klxx_lam0_theta0_equals_mean_z():
    z_pi = jnp.array([0.2, -1.0, 3.5, 0.8])
    z_xi = jnp.array([9.0, -4.0, 1.1, 0.0])
    got = klxx_loss(z_pi, z_xi, lam=0.0, theta=0.0)
    assert jnp.allclose(got, jnp.mean(z_pi), atol=1e-6)


def test_identity_flow_z_is_u0_minus_u():
    key = jax.random.PRNGKey(0)
    params = init_coupling(key, dim=2, n_layers=4, hidden=8)
    y = jnp.array([[0.3, -0.4], [1.2, 0.1], [-1.5, 0.8]])
    gy, logdet = forward(params, y)
    assert jnp.allclose(gy, y, atol=1e-5)
    assert jnp.allclose(logdet, jnp.zeros((3,)), atol=1e-5)
    z = z_from_flow(params, y, two_mode_potential, source_potential, forward)
    u0 = jax.vmap(source_potential)(y)
    u = jax.vmap(two_mode_potential)(y)
    assert jnp.allclose(z, u0 - u, atol=1e-5)
    assert jnp.allclose(z, log_ratio_z(u0, u, logdet), atol=1e-5)
