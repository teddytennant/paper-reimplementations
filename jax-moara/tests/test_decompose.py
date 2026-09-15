import jax.numpy as jnp
import numpy as np
from jax import random

from moara.decompose import decompose, recompose, repeat_blocks


def _rel_err(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(a), 1e-30))


def test_recompose_no_pad():
    key = random.PRNGKey(0)
    g = random.normal(key, (8, 16))
    b = 4
    m, v = decompose(g, b)
    g_hat = recompose(m, v, b, n=g.shape[1])
    assert g_hat.shape == g.shape
    assert _rel_err(g, g_hat) < 1e-5


def test_recompose_with_pad():
    key = random.PRNGKey(1)
    g = random.normal(key, (7, 10))
    b = 4
    assert g.shape[1] % b != 0
    m, v = decompose(g, b)
    assert v.shape == g.shape
    assert m.shape == (7, 3)
    g_hat = recompose(m, v, b, n=g.shape[1])
    assert g_hat.shape == g.shape
    assert _rel_err(g, g_hat) < 1e-5


def test_direction_blocks_unit_norm():
    key = random.PRNGKey(2)
    g = random.normal(key, (6, 12))
    b = 3
    mag, v = decompose(g, b)
    m, n = g.shape
    k = n // b
    v_blocks = v.reshape(m, k, b)
    g_blocks = g.reshape(m, k, b)
    norms = jnp.linalg.norm(v_blocks, axis=-1)
    g_norms = jnp.linalg.norm(g_blocks, axis=-1)
    mask = g_norms > 1e-8
    np.testing.assert_allclose(np.array(norms[mask]), 1.0, atol=1e-5, rtol=1e-5)
    near_zero = ~mask
    if bool(np.any(near_zero)):
        assert float(jnp.max(norms[near_zero])) < 1e-5


def test_near_zero_blocks_not_unit():
    g = jnp.zeros((3, 8))
    g = g.at[0, :4].set(jnp.array([3.0, 0.0, 4.0, 0.0]))
    mag, v = decompose(g, 4)
    n0 = float(jnp.linalg.norm(v[0, :4]))
    n1 = float(jnp.linalg.norm(v[1, :4]))
    assert abs(n0 - 1.0) < 1e-5
    assert n1 < 1e-6


def test_block_size_one():
    key = random.PRNGKey(3)
    g = random.normal(key, (5, 9))
    m, v = decompose(g, 1)
    assert m.shape == (5, 9)
    assert v.shape == (5, 9)
    g_hat = recompose(m, v, 1)
    assert g_hat.shape == g.shape
    assert _rel_err(g, g_hat) < 1e-5


def test_block_size_equals_n():
    key = random.PRNGKey(4)
    g = random.normal(key, (5, 9))
    n = g.shape[1]
    m, v = decompose(g, n)
    assert m.shape == (5, 1)
    assert v.shape == g.shape
    row_norms = jnp.linalg.norm(v, axis=1)
    np.testing.assert_allclose(np.array(row_norms), 1.0, atol=1e-5)
    g_hat = recompose(m, v, n)
    assert g_hat.shape == g.shape
    assert _rel_err(g, g_hat) < 1e-5


def test_repeat_blocks_crops():
    m = jnp.arange(6.0).reshape(2, 3)
    rep = repeat_blocks(m, 4, n=10)
    assert rep.shape == (2, 10)
    assert float(rep[0, 0]) == float(rep[0, 3]) == 0.0
    assert float(rep[0, 4]) == 1.0
    assert float(rep[0, 8]) == 2.0
    assert float(rep[0, 9]) == 2.0
