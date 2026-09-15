import jax.numpy as jnp
import numpy as np
from jax import random

from moara.optim import moara_update


def test_moara_update_changes_w_and_lowrank_state():
    key = random.PRNGKey(0)
    k_w, k_g = random.split(key)
    w = random.normal(k_w, (16, 24))
    g = random.normal(k_g, (16, 24))
    r = 4
    w_new, state = moara_update(w, g, None, block_size=8, rank=r, lr=1e-2)
    assert w_new.shape == w.shape
    assert float(jnp.linalg.norm(w_new - w)) > 0.0
    assert state.vt_mean.shape[0] == r
    assert state.vt_mean.shape[0] != w.shape[0]
    assert state.P.shape == (16, r)
    assert state.m_mean.shape[1] == 24 // 8


def test_zero_gradient_noop():
    key = random.PRNGKey(1)
    w = random.normal(key, (8, 12))
    g = jnp.zeros_like(w)
    w_new, _state = moara_update(w, g, None, block_size=4, rank=3, lr=1e-2)
    np.testing.assert_allclose(np.array(w_new), np.array(w), atol=1e-10)


def test_block_size_one_and_n_in_update():
    key = random.PRNGKey(2)
    k_w, k_g = random.split(key)
    w = random.normal(k_w, (6, 10))
    g = random.normal(k_g, (6, 10))
    w1, s1 = moara_update(w, g, None, block_size=1, rank=2, lr=1e-2)
    assert w1.shape == w.shape
    assert s1.vt_mean.shape == (2, 10)
    wn, sn = moara_update(w, g, None, block_size=10, rank=2, lr=1e-2)
    assert wn.shape == w.shape
    assert sn.m_mean.shape == (6, 1)
    assert sn.vt_mean.shape[0] == 2


def test_pad_columns_in_update():
    key = random.PRNGKey(3)
    k_w, k_g = random.split(key)
    w = random.normal(k_w, (5, 10))
    g = random.normal(k_g, (5, 10))
    w_new, state = moara_update(w, g, None, block_size=4, rank=2, lr=1e-2)
    assert w_new.shape == (5, 10)
    assert state.m_mean.shape == (5, 3)
    assert state.vt_mean.shape == (2, 10)
