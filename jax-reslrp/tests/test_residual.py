"""LRP-γ / LRP-0 residual rules: conservation, bounds, γ=0, batching."""

import jax.numpy as jnp
from jax import random

from reslrp.measures import relevance_amplification
from reslrp.residual import lrp0, lrp_gamma, phi


def test_conservation_lrp_gamma():
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)
    z_in = random.normal(k1, (16, 8))
    z_up = random.normal(k2, (16, 8))
    r_out = random.normal(k3, (16, 8))
    for gamma in (0.1, 0.5, 1.0, 5.0):
        r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma)
        assert jnp.allclose(r_in + r_up, r_out, rtol=1e-10, atol=1e-10)


def test_conservation_lrp0_away_from_zero():
    key = random.PRNGKey(1)
    k1, k2, k3 = random.split(key, 3)
    z_in = random.normal(k1, (32, 8))
    z_up = random.normal(k2, (32, 8))
    r_out = random.normal(k3, (32, 8))
    z_out = z_in + z_up
    r_in, r_up = lrp0(z_in, z_up, r_out, eps=0.0)
    err = jnp.abs(r_in + r_up - r_out)
    assert jnp.all(err[jnp.abs(z_out) > 1e-3] < 1e-8)


def test_gamma_zero_matches_lrp0():
    key = random.PRNGKey(2)
    k1, k2, k3 = random.split(key, 3)
    z_in = random.normal(k1, (16, 8))
    z_up = random.normal(k2, (16, 8))
    r_out = random.normal(k3, (16, 8))
    z_out = z_in + z_up
    mask = jnp.abs(z_out) > 1e-3
    r0_in, r0_up = lrp0(z_in, z_up, r_out, eps=0.0)
    rg_in, rg_up = lrp_gamma(z_in, z_up, r_out, gamma=0.0, eps=1e-12)
    assert jnp.allclose(rg_in[mask], r0_in[mask], rtol=1e-10, atol=1e-10)
    assert jnp.allclose(rg_up[mask], r0_up[mask], rtol=1e-10, atol=1e-10)


def test_same_sign_shares():
    z_in = jnp.asarray(3.0)
    z_up = jnp.asarray(1.0)
    r_out = jnp.asarray(4.0)
    for gamma in (0.0, 0.5, 1.0, 5.0):
        r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma)
        assert jnp.allclose(r_in, 0.75 * r_out)
        assert jnp.allclose(r_up, 0.25 * r_out)
    r0_in, r0_up = lrp0(z_in, z_up, r_out)
    assert jnp.allclose(r0_in, 0.75 * r_out)
    assert jnp.allclose(r0_up, 0.25 * r_out)


def test_opposite_sign_near_cancellation_bound():
    z_in = jnp.asarray(1.0)
    z_up = jnp.asarray(-0.99)
    r_out = jnp.asarray(1.0)
    r0_in, r0_up = lrp0(z_in, z_up, r_out, eps=0.0)
    assert float(jnp.maximum(jnp.abs(r0_in), jnp.abs(r0_up))) > 10.0

    gamma = 1.0
    r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma=gamma)
    a = relevance_amplification(r_in, r_up, r_out)
    assert float(a) <= 1.0 + 2.0 / gamma + 1e-9
    ph = phi(z_in, z_up, gamma)
    lo, hi = -1.0 / gamma, 1.0 + 1.0 / gamma
    assert float(lo) - 1e-9 <= float(ph) <= float(hi) + 1e-9


def test_phi_bound_random():
    key = random.PRNGKey(3)
    k1, k2 = random.split(key)
    z_in = random.normal(k1, (256,))
    z_up = random.normal(k2, (256,))
    z_out = z_in + z_up
    gamma = 0.5
    ph = phi(z_in, z_up, gamma)
    lo, hi = -1.0 / gamma, 1.0 + 1.0 / gamma
    ok = jnp.abs(z_out) > 1e-8
    assert jnp.all(ph[ok] >= lo - 1e-8)
    assert jnp.all(ph[ok] <= hi + 1e-8)
    r_out = jnp.ones_like(z_in)
    r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma)
    # Eq. 5 is elementwise; check per-site A on stable sites
    site_a = (jnp.abs(r_in) + jnp.abs(r_up)) / jnp.maximum(jnp.abs(r_out), 1e-30)
    assert jnp.all(site_a[ok] <= 1.0 + 2.0 / gamma + 1e-6)


def test_zero_r_out():
    z_in = jnp.array([1.0, -2.0, 0.3])
    z_up = jnp.array([-0.5, 0.1, 1.2])
    r_out = jnp.zeros_like(z_in)
    r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma=1.0)
    assert jnp.allclose(r_in, 0.0)
    assert jnp.allclose(r_up, 0.0)
    r0_in, r0_up = lrp0(z_in, z_up, r_out)
    assert jnp.allclose(r0_in, 0.0)
    assert jnp.allclose(r0_up, 0.0)


def test_batched_shapes():
    key = random.PRNGKey(4)
    k1, k2, k3 = random.split(key, 3)
    z_in = random.normal(k1, (5, 16))
    z_up = random.normal(k2, (5, 16))
    r_out = random.normal(k3, (5, 16))
    r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma=1.0)
    assert r_in.shape == (5, 16)
    assert r_up.shape == (5, 16)
    r0_in, r0_up = lrp0(z_in, z_up, r_out)
    assert r0_in.shape == (5, 16)
    assert r0_up.shape == (5, 16)
