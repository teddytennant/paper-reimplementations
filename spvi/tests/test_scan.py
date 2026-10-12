"""Affine scan, BlockScan, and query checks against explicit loops."""

import jax
import jax.numpy as jnp

from spvi.flow import flow
from spvi.scan import affine_prefix_scan, affine_star, block_scan, query_states


def _hand_coeff(j, marks, lower_pre, lower_post):
    """Block-triangular coefficients that never read block j's own state."""
    if j == 0:
        retention = 0.3 + 0.2 * jnp.tanh(marks[..., :1])
        shift = 0.5 * marks[..., :1]
        return retention, shift
    retention = 0.4 + 0.1 * jnp.tanh(lower_post + marks[..., :1])
    shift = 0.25 * lower_pre - 0.1 * marks[..., 1:2]
    return retention, shift


def _chronological(z0, times, marks, gates, gamma, pad):
    """Independent loop of the block-triangular recurrence, padding as identity."""
    batch, length = times.shape
    rows = []
    for b in range(batch):
        z = z0[b]
        prev_t = jnp.asarray(0.0, dtype=times.dtype)
        stored = []
        for i in range(length):
            if bool(pad[b, i]):
                stored.append(z)
                continue
            delta = times[b, i] - prev_t
            z_pre = jnp.exp(-gamma * delta) * z
            g = gates[b, i]
            mark = marks[b, i]
            r0 = 0.3 + 0.2 * jnp.tanh(mark[0])
            b0 = 0.5 * mark[0]
            z0_post = (1.0 + g * (r0 - 1.0)) * z_pre[0] + g * b0
            r1 = 0.4 + 0.1 * jnp.tanh(z0_post + mark[0])
            b1 = 0.25 * z_pre[0] - 0.1 * mark[1]
            z1_post = (1.0 + g * (r1 - 1.0)) * z_pre[1] + g * b1
            z = jnp.stack([z0_post, z1_post])
            stored.append(z)
            prev_t = times[b, i]
        rows.append(jnp.stack(stored, axis=0))
    post = jnp.stack(rows, axis=0)
    return jnp.concatenate([z0[:, None, :], post], axis=1)


def test_affine_star_associative_identity_and_scan():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    a1 = jax.random.uniform(k1, (4,), minval=0.2, maxval=1.4)
    b1 = jax.random.normal(k1, (4,))
    a2 = jax.random.uniform(k2, (4,), minval=0.2, maxval=1.4)
    b2 = jax.random.normal(k2, (4,))
    a3 = jax.random.uniform(k3, (4,), minval=0.2, maxval=1.4)
    b3 = jax.random.normal(k3, (4,))

    left = affine_star((a3, b3), affine_star((a2, b2), (a1, b1)))
    right = affine_star(affine_star((a3, b3), (a2, b2)), (a1, b1))
    assert jnp.allclose(left[0], right[0], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(left[1], right[1], atol=1e-5, rtol=1e-5)

    ones = jnp.ones_like(a1)
    zeros = jnp.zeros_like(b1)
    id_right = affine_star((a1, b1), (ones, zeros))
    id_left = affine_star((ones, zeros), (a1, b1))
    assert jnp.allclose(id_right[0], a1) and jnp.allclose(id_right[1], b1)
    assert jnp.allclose(id_left[0], a1) and jnp.allclose(id_left[1], b1)

    length, dim = 9, 3
    ka, kb = jax.random.split(jax.random.PRNGKey(1))
    a = jax.random.uniform(ka, (length, dim), minval=0.3, maxval=1.2)
    b = jax.random.normal(kb, (length, dim))
    # Identity padding, including a middle slot and a tail.
    pad_idx = jnp.asarray([3, 7, 8])
    a = a.at[pad_idx].set(1.0)
    b = b.at[pad_idx].set(0.0)

    a_pref, b_pref = affine_prefix_scan(a, b, axis=0)
    acc_a, acc_b = a[0], b[0]
    seq_a = [acc_a]
    seq_b = [acc_b]
    for i in range(1, length):
        acc_a, acc_b = affine_star((a[i], b[i]), (acc_a, acc_b))
        seq_a.append(acc_a)
        seq_b.append(acc_b)
    seq_a = jnp.stack(seq_a)
    seq_b = jnp.stack(seq_b)
    assert jnp.allclose(a_pref, seq_a, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(b_pref, seq_b, atol=1e-5, rtol=1e-5)
    # Identity padding does not change the preceding prefix.
    assert jnp.allclose(a_pref[3], a_pref[2], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(b_pref[3], b_pref[2], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(a_pref[8], a_pref[6], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(b_pref[8], b_pref[6], atol=1e-5, rtol=1e-5)


def test_block_scan_matches_chronological_and_queries():
    key = jax.random.PRNGKey(2)
    kz, kt, km, kg = jax.random.split(key, 4)
    batch, length, dim = 2, 6, 2
    z0 = jax.random.normal(kz, (batch, dim))
    gamma = jnp.asarray([0.4, 0.7])
    # Real events, then a padding tail.
    real = jnp.sort(jax.random.uniform(kt, (batch, 4)) * 3.0 + 0.1, axis=1)
    pad_times = jnp.asarray([10.0, 11.0]) + jnp.arange(batch)[:, None] * 0.0
    times = jnp.concatenate([real, jnp.broadcast_to(pad_times, (batch, 2))], axis=1)
    marks = jax.random.normal(km, (batch, length, 2))
    gates = (jax.random.uniform(kg, (batch, length)) > 0.4).astype(jnp.float32)
    pad = jnp.zeros((batch, length), dtype=bool).at[:, 4:].set(True)
    gates = jnp.where(pad, 0.0, gates)
    # The tie query is at the second event; force a jump so pre and post differ.
    gates = gates.at[:, 1].set(1.0)

    states = block_scan(
        z0, times, marks, gates, gamma, _hand_coeff, (1, 1), pad_mask=pad
    )
    reference = _chronological(z0, times, marks, gates, gamma, pad)
    assert jnp.allclose(states, reference, atol=1e-5, rtol=1e-5)
    # Padding tail equals the last real post-event state.
    assert jnp.allclose(states[:, 5], states[:, 4], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(states[:, 6], states[:, 4], atol=1e-5, rtol=1e-5)

    # L = 0 path is the pure flow.
    empty_t = jnp.zeros((batch, 0))
    empty_m = jnp.zeros((batch, 0, 2))
    empty_g = jnp.zeros((batch, 0))
    states0 = block_scan(z0, empty_t, empty_m, empty_g, gamma, _hand_coeff, (1, 1))
    query_t = jnp.asarray([0.0, 0.5, 1.7, 3.0])
    got0 = query_states(states0, empty_t, query_t, gamma)
    expect0 = flow(z0[:, None, :], query_t, gamma)
    assert jnp.allclose(got0, expect0, atol=1e-5, rtol=1e-5)

    # Between events, the query is the flow from the preceding post-event state.
    mid = 0.5 * (times[:, 0] + times[:, 1])
    queried = query_states(states, times, mid[:, None], gamma, valid=~pad)
    expected = flow(states[:, 1, :], mid - times[:, 0], gamma)
    assert jnp.allclose(queried[:, 0], expected, atol=1e-5, rtol=1e-5)

    # Exact tie: observations use the post-event state, KL probes the pre-event state.
    tie = times[:, 1]
    post_q = query_states(states, times, tie[:, None], gamma, valid=~pad, pre_event_on_tie=False)
    pre_q = query_states(states, times, tie[:, None], gamma, valid=~pad, pre_event_on_tie=True)
    gap = times[:, 1] - times[:, 0]
    pre = flow(states[:, 1, :], gap, gamma)
    assert jnp.allclose(post_q[:, 0], states[:, 2], atol=1e-5, rtol=1e-5)
    assert jnp.allclose(pre_q[:, 0], pre, atol=1e-5, rtol=1e-5)
    # Gates are not all zero, so at least one series jumps and pre != post.
    assert jnp.any(jnp.abs(pre - states[:, 2]) > 1e-3)


def test_coefficients_ignore_own_and_upper_blocks():
    key = jax.random.PRNGKey(3)
    kz, kt, km, kg = jax.random.split(key, 4)
    batch, length = 2, 5
    z0 = jax.random.normal(kz, (batch, 2))
    times = jnp.sort(jax.random.uniform(kt, (batch, length)) * 2.0 + 0.05, axis=1)
    marks = jax.random.normal(km, (batch, length, 2))
    gates = (jax.random.uniform(kg, (batch, length)) > 0.5).astype(jnp.float32)
    gamma = jnp.asarray([0.3, 0.8])
    pad = jnp.zeros((batch, length), dtype=bool)

    base = block_scan(z0, times, marks, gates, gamma, _hand_coeff, (1, 1), pad_mask=pad)
    # Upper-block initial state must not change the lower-block trajectory.
    z0_upper = z0.at[:, 1].add(1.7)
    shifted = block_scan(z0_upper, times, marks, gates, gamma, _hand_coeff, (1, 1), pad_mask=pad)
    assert jnp.allclose(base[..., 0], shifted[..., 0], atol=1e-5, rtol=1e-5)
    assert jnp.any(jnp.abs(base[..., 1] - shifted[..., 1]) > 1e-3)

    # Block-1 coefficients change with the lower post-event state, not with a
    # fake own-state channel (the coefficient function has no such input).
    lower_pre = jnp.asarray([0.2, -0.4])[None, None, :]
    lower_post = jnp.asarray([0.5, 0.1])[None, None, :]
    mark = jnp.asarray([0.3, -0.2])[None, None, :]
    r_a, b_a = _hand_coeff(1, mark, lower_pre[..., :1], lower_post[..., :1])
    r_b, b_b = _hand_coeff(1, mark, lower_pre[..., :1], lower_post[..., :1] + 0.8)
    assert jnp.allclose(r_a, 0.4 + 0.1 * jnp.tanh(lower_post[..., :1] + mark[..., :1]))
    assert jnp.allclose(b_a, 0.25 * lower_pre[..., :1] - 0.1 * mark[..., 1:2])
    assert jnp.any(jnp.abs(r_a - r_b) > 1e-4)
    # Own pre-event state is not an argument; a different own value cannot enter.
    own_a = 3.0
    own_b = -2.0
    assert own_a != own_b
    r_same, b_same = _hand_coeff(1, mark, lower_pre[..., :1], lower_post[..., :1])
    assert jnp.allclose(r_same, r_a) and jnp.allclose(b_same, b_a)

    reference = _chronological(z0, times, marks, gates, gamma, pad)
    assert jnp.allclose(base, reference, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(shifted[..., 1], _chronological(z0_upper, times, marks, gates, gamma, pad)[..., 1], atol=1e-5, rtol=1e-5)
