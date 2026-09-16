"""LOD query unit tests (Eq. 5, 12--15)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_gralod.query import (
    clamp_lod,
    lod_query,
    lod_query_lerp,
    neighboring_levels,
    triangular_weights,
)


def _levels(seed: int = 0, h: int = 8, c: int = 8, l: int = 3):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, l)
    return [jax.random.normal(k, (1, h, h, c)) for k in keys]


def test_clamp_range():
    lam = jnp.array([-1.0, 0.0, 1.5, 9.0])
    out = clamp_lod(lam, 3)
    np.testing.assert_allclose(out, jnp.array([0.0, 0.0, 1.5, 2.0]))


def test_neighboring_levels_integer_and_fractional():
    lam = jnp.array([0.0, 0.4, 1.0, 1.9, 2.0])
    l0, l1, a = neighboring_levels(lam, 3)
    np.testing.assert_allclose(l0, jnp.array([0.0, 0.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(l1, jnp.array([0.0, 1.0, 1.0, 2.0, 2.0]))
    np.testing.assert_allclose(a, jnp.array([0.0, 0.4, 0.0, 0.9, 0.0]))
