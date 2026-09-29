"""Associative scan agrees with sequential affine stepping (Eq. 4, Prop. B.1)."""

import jax
import jax.numpy as jnp

from slisde.scan import affine_compose, parallel_simulate, sequential_simulate
from slisde.sde import euler_maruyama_pairs, sample_increments


def test_compose_matches_equation_4():
    F = jnp.array([[1.0, 0.2], [0.0, 0.9]])
    g = jnp.array([0.1, -0.3])
    Fp = jnp.array([[0.8, 0.0], [0.1, 1.1]])
    gp = jnp.array([0.0, 0.4])
    F_c, g_c = affine_compose((F, g), (Fp, gp))
    assert jnp.allclose(F_c, Fp @ F)
    assert jnp.allclose(g_c, Fp @ g + gp)


def test_parallel_matches_sequential_random_maps():
    key = jax.random.PRNGKey(0)
    B, N, d = 5, 12, 3
    kF, kg, kz = jax.random.split(key, 3)
    F = jax.random.normal(kF, (B, N, d, d)) * 0.2 + jnp.eye(d)
    g = jax.random.normal(kg, (B, N, d)) * 0.1
    z0 = jax.random.normal(kz, (B, d))
    seq = sequential_simulate(F, g, z0)
    par = parallel_simulate(F, g, z0)
    assert seq.shape == (B, N + 1, d)
    assert jnp.allclose(seq, par, atol=1e-5, rtol=1e-5)


def test_parallel_matches_manual_loop_on_em_layer():
    key = jax.random.PRNGKey(1)
    B, N, d, m = 4, 10, 2, 2
    dt = 0.05
    A = jnp.array([[-0.4, 0.1], [0.0, -0.6]])
    b = jnp.array([0.05, -0.02])
    C = jnp.zeros((m, d, d))
    D = 0.3 * jnp.eye(d, m)
    k0, kW = jax.random.split(key)
    z0 = jax.random.normal(k0, (B, d))
    dW = sample_increments(kW, B, N, m, dt)
    F, g = euler_maruyama_pairs(A, b, C, D, dW, dt)
    par = parallel_simulate(F, g, z0)
    z = z0
    paths = [z0]
    for k in range(N):
        z = jnp.einsum("bij,bj->bi", F[:, k], z) + g[:, k]
        paths.append(z)
    manual = jnp.stack(paths, axis=1)
    assert jnp.allclose(par, manual, atol=1e-5)
    assert jnp.allclose(par, sequential_simulate(F, g, z0), atol=1e-5)
