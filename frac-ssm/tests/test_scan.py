"""Sequential Algorithm 1 matches parallel affine scan (eq. 24)."""

import jax
import jax.numpy as jnp

from frac_ssm.transition import affine_compose, parallel_scan, sequential_scan, transition_sequence


def test_affine_compose_pair():
    rho = jnp.array([0.8, 0.5])
    b = jnp.array([0.1, -0.2])
    rhop = jnp.array([0.3, 1.1])
    bp = jnp.array([0.4, 0.0])
    rho_c, b_c = affine_compose((rho, b), (rhop, bp))
    assert jnp.allclose(rho_c, rhop * rho)
    assert jnp.allclose(b_c, rhop * b + bp)


def test_sequential_equals_parallel_scan():
    key = jax.random.PRNGKey(0)
    B, T, H, M, dh = 3, 16, 2, 4, 5
    k_r, k_b, k_s = jax.random.split(key, 3)
    rho = jax.nn.sigmoid(jax.random.normal(k_r, (B, T, H, M, 1)))
    b = jax.random.normal(k_b, (B, T, H, M, dh)) * 0.2
    s0 = jax.random.normal(k_s, (B, H, M, dh)) * 0.1
    seq = sequential_scan(rho, b, s0)
    par = parallel_scan(rho, b, s0)
    assert seq.shape == (B, T, H, M, dh)
    assert jnp.allclose(seq, par, atol=1e-5, rtol=1e-5)


def test_transition_sequence_loop_equals_scan():
    key = jax.random.PRNGKey(1)
    B, T, H, M, dh = 2, 12, 2, 4, 3
    keys = jax.random.split(key, 7)
    u = jax.random.normal(keys[0], (B, T, H, dh))
    delta = jax.nn.softplus(jax.random.normal(keys[1], (B, T, H)))
    alpha = jax.nn.sigmoid(jax.random.normal(keys[2], (B, T, H)))
    lam = jax.nn.softplus(jax.random.normal(keys[3], (B, T, H)))
    tau = jnp.array([0.5, 1.0, 2.0, 4.0])
    g_read = jax.random.normal(keys[4], (B, T, H, M)) * 0.3
    g_write = jax.random.normal(keys[5], (B, T, H, M)) * 0.3
    s0 = jax.random.normal(keys[6], (B, H, M, dh)) * 0.05
    h_seq, S_seq = transition_sequence(
        u, delta, alpha, lam, tau, g_read, g_write, s0=s0, parallel=False
    )
    h_par, S_par = transition_sequence(
        u, delta, alpha, lam, tau, g_read, g_write, s0=s0, parallel=True
    )
    assert jnp.allclose(S_seq, S_par, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(h_seq, h_par, atol=1e-5, rtol=1e-5)
