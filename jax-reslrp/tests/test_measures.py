"""Eq. 2 cancellation C and amplification A."""

import jax.numpy as jnp

from reslrp.measures import amplification, cancellation, relevance_amplification
from reslrp.residual import lrp_gamma


def test_cancellation_same_sign_is_one():
    # Eq. 2: C = 1 means no cancellation.
    c = cancellation(jnp.asarray(3.0), jnp.asarray(1.0))
    assert jnp.allclose(c, 1.0)
    c_vec = cancellation(jnp.array([2.0, 4.0]), jnp.array([1.0, 1.0]))
    assert jnp.allclose(c_vec, 1.0)


def test_cancellation_opposite_is_large():
    c = cancellation(jnp.asarray(1.0), jnp.asarray(-1.0 + 1e-6))
    assert float(c) > 10.0
    c_exact = cancellation(jnp.asarray(1.0), jnp.asarray(-1.0), eps=1e-12)
    assert float(c_exact) > 1e6


def test_amplification_no_update_is_one():
    z_in = jnp.array([1.5, -0.25, 3.0])
    z_out = z_in  # z_up = 0
    assert jnp.allclose(amplification(z_in, z_out), 1.0)


def test_amplification_large_under_cancellation():
    z_in = jnp.asarray(1.0)
    z_up = jnp.asarray(-0.99)
    z_out = z_in + z_up
    a = amplification(z_in, z_out)
    assert float(a) > 10.0


def test_relevance_amplification_bounded():
    z_in = jnp.asarray(1.0)
    z_up = jnp.asarray(-0.99)
    r_out = jnp.asarray(1.0)
    gamma = 1.0
    r_in, r_up = lrp_gamma(z_in, z_up, r_out, gamma)
    a = relevance_amplification(r_in, r_up, r_out)
    assert float(a) <= 1.0 + 2.0 / gamma + 1e-9
