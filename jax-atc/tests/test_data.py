"""Curriculum sampler: frozen suffix, varying prefix, matching parity target."""

import jax
import jax.numpy as jnp

from jax_atc.data import parity_target, sample_parity, sample_subset


def test_curriculum_sampler_prefix():
    n, r, B = 8, 3, 64
    bits, y = sample_parity(jax.random.PRNGKey(0), n, r, B)
    assert bits.shape == (B, n)
    assert y.shape == (B,)
    assert jnp.all(bits[:, r:] == 1.0)
    assert jnp.all(jnp.isin(bits[:, :r], jnp.array([-1.0, 1.0])))
    assert jnp.allclose(y, jnp.prod(bits[:, :r], axis=1))
    # first r positions actually vary
    assert jnp.any(bits[:, 0] == 1.0) and jnp.any(bits[:, 0] == -1.0)


def test_curriculum_sampler_subset():
    n, r, B = 8, 5, 32
    subset = jnp.array([0, 2, 6])
    bits, y = sample_parity(jax.random.PRNGKey(1), n, r, B, subset=subset)
    assert jnp.all(bits[:, r:] == 1.0)
    expect = parity_target(bits, r, subset=subset)
    assert jnp.allclose(y, expect)
    # S ∩ {1..r} = {0, 2}; position 6 is frozen at +1 so it does not affect y
    assert jnp.allclose(y, bits[:, 0] * bits[:, 2])


def test_sample_subset_size():
    s = sample_subset(jax.random.PRNGKey(2), n=10, k=4)
    assert s.shape == (4,)
    assert len(jnp.unique(s)) == 4
