"""Checks for per-head QCOC compression."""

import jax.numpy as jnp
from jax.nn import softmax

from qcoc import attend, compress
from qcoc.compress import phi


def _spread(n, d, dv):
    idx = jnp.arange(n, dtype=jnp.float32)
    key_cols = [
        jnp.sin(idx * 0.7 + 0.2),
        jnp.cos(idx * 0.5),
        jnp.sin(idx * 1.3 + 1.0),
    ]
    value_cols = [
        jnp.cos(idx * 0.4 + 0.3),
        jnp.sin(idx * 0.9),
        idx * 0.15 - 0.4,
    ]
    query_cols = [
        jnp.sin(idx * 0.35 + 0.4),
        jnp.cos(idx * 0.8 + 0.1),
        idx * 0.05,
    ]
    keys = jnp.stack(key_cols, axis=-1)[:, :d]
    values = jnp.stack(value_cols, axis=-1)[:, :dv]
    queries = jnp.stack(query_cols, axis=-1)[:, :d]
    return keys, values, queries


def _full_attention(queries, keys, values):
    return softmax(phi(queries) @ keys.T, axis=-1) @ values


def test_weights_sum_and_cover():
    keys, values, queries = _spread(10, 3, 2)
    out = compress(keys, values, queries, n_prototypes=4)
    weights = out["weights"]
    assert int(jnp.sum(weights)) == 10
    assert bool(jnp.all(weights >= 1))
    assert jnp.allclose(out["log_w"], jnp.log(weights.astype(jnp.float32)))


def test_identical_keys_values_match_full_attention():
    keys = jnp.full((6, 2), 0.3)
    values = jnp.full((6, 2), -0.7)
    queries = jnp.array(
        [
            [0.0, 0.1],
            [1.2, -0.4],
            [-0.5, 0.8],
            [0.2, 0.2],
            [2.0, 0.0],
            [-1.0, -0.3],
        ],
        dtype=jnp.float32,
    )
    memory = compress(keys, values, queries, n_prototypes=3)
    pred = attend(queries, memory)
    full = _full_attention(queries, keys, values)
    assert jnp.allclose(pred, full, atol=1e-5)


def test_fitting_residual_does_not_increase():
    keys, values, queries = _spread(9, 3, 2)
    out = compress(keys, values, queries, n_prototypes=3)
    err_star = jnp.linalg.norm(out["P"] @ out["values"] - out["Y"])
    err_anchor = jnp.linalg.norm(out["P"] @ out["anchor_values"] - out["Y"])
    assert float(err_star) <= float(err_anchor) + 1e-5


def test_vstar_matches_hand_solve():
    keys = jnp.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.1, 0.15],
            [4.0, 3.5],
            [4.2, 3.4],
            [3.8, 3.7],
        ],
        dtype=jnp.float32,
    )
    values = jnp.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.1, -0.05],
            [-0.2, 2.0],
            [0.0, 1.8],
            [-0.1, 2.1],
        ],
        dtype=jnp.float32,
    )
    queries = jnp.array(
        [
            [0.1, 0.0],
            [0.0, 0.2],
            [1.0, 0.5],
            [3.5, 3.0],
            [4.0, 4.0],
            [2.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    ridge = 1e-3
    eps_num = 1e-12
    out = compress(keys, values, queries, n_prototypes=2, ridge=ridge, eps_num=eps_num)

    group0 = jnp.array([0, 1, 2])
    group1 = jnp.array([3, 4, 5])
    assert jnp.allclose(out["keys"][0], jnp.mean(keys[group0], axis=0), atol=1e-5)
    assert jnp.allclose(out["keys"][1], jnp.mean(keys[group1], axis=0), atol=1e-5)
    assert jnp.allclose(out["anchor_values"][0], jnp.mean(values[group0], axis=0), atol=1e-5)
    assert jnp.allclose(out["anchor_values"][1], jnp.mean(values[group1], axis=0), atol=1e-5)
    assert jnp.array_equal(out["weights"], jnp.array([3, 3], dtype=out["weights"].dtype))

    y_hand = _full_attention(queries, keys, values)
    assert jnp.allclose(out["Y"], y_hand, atol=1e-5)
    p_hand = softmax(phi(queries) @ out["keys"].T + out["log_w"][None, :], axis=-1)
    assert jnp.allclose(out["P"], p_hand, atol=1e-5)

    p = out["P"]
    y = out["Y"]
    v0 = out["anchor_values"]
    gamma = jnp.maximum(jnp.sum(p * p) / 2, eps_num)
    gram = p.T @ p + ridge * gamma * jnp.eye(2, dtype=p.dtype)
    rhs = p.T @ y + ridge * gamma * v0
    det = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
    inv = jnp.array(
        [[gram[1, 1], -gram[0, 1]], [-gram[1, 0], gram[0, 0]]],
        dtype=gram.dtype,
    ) / det
    expected = inv @ rhs
    assert jnp.allclose(out["values"], expected, atol=1e-5, rtol=1e-5)


def test_deterministic():
    keys, values, queries = _spread(8, 3, 2)
    first = compress(keys, values, queries, n_prototypes=3)
    second = compress(keys, values, queries, n_prototypes=3)
    assert first["n_examples"] == second["n_examples"]
    for name in ("keys", "values", "log_w", "weights", "anchor_values", "P", "Y"):
        assert jnp.array_equal(first[name], second[name])


def test_compression_ratio_four():
    keys, values, queries = _spread(8, 2, 2)
    out = compress(keys, values, queries, n_prototypes=2)
    assert out["n_examples"] == 8
    assert out["keys"].shape[0] == 2
    assert out["values"].shape[0] == 2
    assert out["n_examples"] / out["keys"].shape[0] == 4
