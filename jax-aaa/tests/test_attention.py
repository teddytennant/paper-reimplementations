"""Axis restriction, dense softmax, and k-NN spatial mask."""

import jax
import jax.numpy as jnp
import pytest

from aaa.attention import (
    dense_attention,
    init_attention_params,
    knn_spatial_mask,
    spatial_attention,
    spatial_token_mask,
    temporal_attention,
    temporal_token_mask,
)


def _grid(key, b=2, c=4, t=5, d=8):
    return jax.random.normal(key, (b, c, t, d))


def test_temporal_mask_same_channel_only():
    mask = temporal_token_mask(4, 5)
    assert mask.shape == (4, 5, 4, 5)
    c_q = jnp.arange(4)[:, None, None, None]
    c_k = jnp.arange(4)[None, None, :, None]
    assert bool(jnp.all(mask == (c_q == c_k)))


def test_spatial_mask_same_time_only():
    mask = spatial_token_mask(4, 5)
    t_q = jnp.arange(5)[None, :, None, None]
    t_k = jnp.arange(5)[None, None, None, :]
    assert bool(jnp.all(mask == (t_q == t_k)))


def test_temporal_attention_zero_across_channels():
    key = jax.random.PRNGKey(0)
    x = _grid(key)
    params = init_attention_params(jax.random.PRNGKey(1), d=8)
    _, weights = temporal_attention(params, x, num_heads=2, return_weights=True)
    b, h, c, t, c2, t2 = weights.shape
    assert (b, h, c, t, c2, t2) == (2, 2, 4, 5, 4, 5)
    cq = jnp.arange(c)[:, None, None, None]
    ck = jnp.arange(c)[None, None, :, None]
    off_mask = cq != ck
    off_mass = jnp.sum(jnp.abs(weights * off_mask[None, None, ...]))
    on_mask = cq == ck
    on_mass = jnp.sum(weights * on_mask[None, None, ...])
    assert float(off_mass) < 1e-6
    # each query's weights over its own channel's T keys sum to 1
    row = jnp.sum(weights, axis=(-2, -1))
    assert jnp.allclose(row, 1.0, atol=1e-5)
    assert float(on_mass) == pytest.approx(float(jnp.sum(row)), rel=1e-5)


def test_spatial_attention_zero_across_times():
    key = jax.random.PRNGKey(2)
    x = _grid(key)
    params = init_attention_params(jax.random.PRNGKey(3), d=8)
    _, weights = spatial_attention(params, x, num_heads=2, return_weights=True)
    tq = jnp.arange(5)[None, :, None, None]
    tk = jnp.arange(5)[None, None, None, :]
    off_mask = tq != tk
    off_mass = jnp.sum(jnp.abs(weights * off_mask[None, None, ...]))
    assert float(off_mass) < 1e-6
    row = jnp.sum(weights, axis=(-2, -1))
    assert jnp.allclose(row, 1.0, atol=1e-5)


def test_dense_attends_to_all_ct_tokens():
    key = jax.random.PRNGKey(4)
    x = _grid(key, b=1, c=3, t=4, d=8)
    params = init_attention_params(jax.random.PRNGKey(5), d=8)
    out, weights = dense_attention(params, x, num_heads=2, return_weights=True)
    assert out.shape == (1, 3, 4, 8)
    assert weights.shape == (1, 2, 3, 4, 3, 4)
    row = jnp.sum(weights, axis=(-2, -1))
    assert jnp.allclose(row, 1.0, atol=1e-5)
    # every key location receives some mass from at least one query/head
    key_mass = jnp.sum(weights, axis=(0, 1, 2, 3))
    assert bool(jnp.all(key_mass > 0))
    # softmax is over CT, so a typical query puts mass on off-axis keys
    cq = jnp.arange(3)[:, None, None, None]
    ck = jnp.arange(3)[None, None, :, None]
    tq = jnp.arange(4)[None, :, None, None]
    tk = jnp.arange(4)[None, None, None, :]
    off_axis = (cq != ck) & (tq != tk)
    off_mass = jnp.sum(weights * off_axis[None, None, ...])
    assert float(off_mass) > 0.1


def test_knn_spatial_mask_self_and_k_nearest():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
        ]
    )
    mask = knn_spatial_mask(positions, k=1)
    assert mask.shape == (4, 4)
    assert bool(jnp.all(jnp.diag(mask)))
    # electrode 0 nearest is 1; 2 nearest is 3
    assert bool(mask[0, 1]) and not bool(mask[0, 2]) and not bool(mask[0, 3])
    assert bool(mask[2, 3]) and not bool(mask[2, 0]) and not bool(mask[2, 1])
    assert int(jnp.sum(mask, axis=-1).min()) == 2
    assert int(jnp.sum(mask, axis=-1).max()) == 2


def test_knn_none_keeps_all_electrodes():
    positions = jnp.zeros((5, 3))
    mask = knn_spatial_mask(positions, k=None)
    assert bool(jnp.all(mask))


def test_spatial_attention_respects_knn_mask():
    key = jax.random.PRNGKey(6)
    c, t, d = 4, 3, 8
    x = jax.random.normal(key, (1, c, t, d))
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
        ]
    )
    knn = knn_spatial_mask(positions, k=1)
    params = init_attention_params(jax.random.PRNGKey(7), d=d)
    _, weights = spatial_attention(
        params, x, num_heads=2, knn_mask=knn, return_weights=True
    )
    # weights[..., cq, tq, ck, tk]: mass only if tq==tk and knn[cq, ck]
    allowed = knn[:, None, :, None] & spatial_token_mask(c, t)
    forbidden = ~allowed
    off_mass = jnp.sum(jnp.abs(weights * forbidden[None, None, ...]))
    assert float(off_mass) < 1e-6
    row = jnp.sum(weights, axis=(-2, -1))
    assert jnp.allclose(row, 1.0, atol=1e-5)
