"""Checks for last-hidden injection, the selective scan, and the drafter."""

import jax
import jax.numpy as jnp
import pytest

from hspec.inject import drafter_kv_bytes, s_init, scan_parallel, scan_sequential
from hspec import inject as inject_mod
from hspec.draft import (
    D,
    D_HEAD,
    D_STATE,
    DEFAULT_GAMMA,
    N_HEADS,
    N_LAYERS,
    VOCAB,
    acceptance_length,
    accepted_prefix,
    apply_rope,
    block_positions,
    count_params,
    draft_qkv,
    drafter_forward,
    init_drafter,
    init_target,
    mask_inputs,
    target_greedy_tokens,
)


def test_s_init_matches_manual_matmul():
    h = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    flat = jnp.ravel(h[..., -1, :])
    proj_r = jnp.arange(5 * 8, dtype=jnp.float32).reshape(5, 8) * 0.1
    proj_ds = jnp.arange(3 * 5, dtype=jnp.float32).reshape(3, 5) * 0.01
    got = s_init(h, proj_r, proj_ds)
    expected = proj_ds @ (proj_r @ flat)
    assert got.shape == (3,)
    assert jnp.allclose(got, expected)

    h0 = jnp.array([[1.0, 2.0, 9.0], [3.0, 4.0, 8.0]], dtype=jnp.float32)
    h1 = jnp.array([[5.0, 6.0, 7.0], [7.0, 8.0, 1.0]], dtype=jnp.float32)
    flat_list = jnp.concatenate([jnp.ravel(h0[-1]), jnp.ravel(h1[-1])])
    proj_r = jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 6) * 0.05
    proj_ds = jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4) * 0.2
    got = s_init((h0, h1), proj_r, proj_ds)
    expected = proj_ds @ (proj_r @ flat_list)
    assert jnp.allclose(got, expected)


def test_s_init_invariant_to_non_last_positions():
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    h = jax.random.normal(k1, (2, 5, 4))
    h_other = h.at[:, :-1, :].set(jax.random.normal(k2, (2, 4, 4)) * 7.0)
    proj_r = jax.random.normal(k3, (6, 8))
    proj_ds = jax.random.normal(k4, (3, 6))
    assert jnp.allclose(s_init(h, proj_r, proj_ds), s_init(h_other, proj_r, proj_ds))

    a = jax.random.normal(k1, (6, 4))
    b = jax.random.normal(k2, (6, 4))
    a2 = a.at[:-1].set(jax.random.normal(k3, (5, 4)) * 3.0)
    b2 = b.at[:-1].set(0.0)
    assert jnp.allclose(s_init([a, b], proj_r, proj_ds), s_init([a2, b2], proj_r, proj_ds))


def test_one_shared_projection_not_per_layer():
    assert list(s_init.__code__.co_varnames[: s_init.__code__.co_argcount]) == [
        "h_last",
        "proj_r",
        "proj_ds",
    ]
    public = [n for n in dir(inject_mod) if not n.startswith("_")]
    assert "proj_r" not in public and "proj_ds" not in public
    for name in public:
        value = getattr(inject_mod, name)
        assert not isinstance(value, (list, tuple))
    h = jnp.ones((N_LAYERS, 4, D), dtype=jnp.float32)
    proj_r = jnp.ones((D_STATE, N_LAYERS * D), dtype=jnp.float32)
    proj_ds = jnp.eye(D_STATE, dtype=jnp.float32)
    out = s_init(h, proj_r, proj_ds)
    assert out.shape == (D_STATE,)
    flat = jnp.ravel(h[..., -1, :])
    assert jnp.allclose(out, proj_ds @ (proj_r @ flat))


def test_scan_parallel_matches_sequential():
    key = jax.random.PRNGKey(1)
    ks = jax.random.split(key, 6)
    x = jax.random.normal(ks[0], (7, 5)) * 0.5
    dt = jax.random.normal(ks[1], (7, 3)) * 0.5
    A = jax.random.normal(ks[2], (3,)) * 0.5
    W_B = jax.random.normal(ks[3], (3, 5)) * 0.5
    x_proj = jax.random.normal(ks[4], (7, 3)) * 0.5
    s_init_state = jax.random.normal(ks[5], (3,))
    assert float(jnp.linalg.norm(s_init_state)) > 0.0
    par = scan_parallel(x, dt, A, W_B, x_proj, s_init_state)
    seq = scan_sequential(x, dt, A, W_B, x_proj, s_init_state)
    assert jnp.allclose(par, seq, atol=1e-5, rtol=1e-5)


def test_different_x_changes_scan_output():
    key = jax.random.PRNGKey(2)
    ks = jax.random.split(key, 5)
    x1 = jax.random.normal(ks[0], (6, 4))
    x2 = x1 + jnp.array([0.7, -0.2, 0.4, 1.1], dtype=jnp.float32)
    dt = jax.random.normal(ks[1], (6, 3)) * 0.3
    A = jax.random.normal(ks[2], (3,)) * 0.3
    W_B = jax.random.normal(ks[3], (3, 4))
    x_proj = jax.random.normal(ks[4], (6, 3))
    s0 = jnp.zeros((3,), dtype=jnp.float32)
    y1 = scan_parallel(x1, dt, A, W_B, x_proj, s0)
    y2 = scan_parallel(x2, dt, A, W_B, x_proj, s0)
    assert y1.shape == y2.shape
    assert not jnp.allclose(y1, y2, atol=1e-5, rtol=1e-5)


def test_drafter_has_no_prefix_kv():
    for prefix_len in (0, 1, 4, 17, 128):
        assert drafter_kv_bytes(prefix_len) == 0
    n = count_params(init_drafter(jax.random.PRNGKey(0)))
    assert n == count_params(init_drafter(jax.random.PRNGKey(1)))
    params = init_drafter(jax.random.PRNGKey(3))
    h = jnp.zeros((N_LAYERS, D), dtype=jnp.float32)
    for prefix_len in (1, 8, 32):
        tk = jax.random.normal(jax.random.PRNGKey(prefix_len), (prefix_len, N_HEADS, D_HEAD))
        tv = jax.random.normal(jax.random.PRNGKey(prefix_len + 9), (prefix_len, N_HEADS, D_HEAD))
        logits = drafter_forward(params, h, tk, tv, prefix_len=prefix_len, gamma=DEFAULT_GAMMA)
        assert logits.shape == (DEFAULT_GAMMA, VOCAB)
        q, k, v = draft_qkv(params, DEFAULT_GAMMA, prefix_len)
        assert q.shape[0] == DEFAULT_GAMMA
        assert k.shape[0] == DEFAULT_GAMMA
        assert v.shape == k.shape
        assert q.shape[0] != prefix_len or prefix_len == DEFAULT_GAMMA
    assert count_params(params) == n
    inputs = mask_inputs(params, 5)
    assert jnp.allclose(inputs[0], inputs[-1])


def test_target_k_changes_draft_logits():
    params = init_drafter(jax.random.PRNGKey(4))
    params = dict(params)
    params["W_q"] = params["W_q"] * 3.0
    params["W_o"] = params["W_o"] * 3.0
    params["lm_head"] = params["lm_head"] * 3.0
    h = jnp.ones((N_LAYERS, D), dtype=jnp.float32)
    prefix = 5
    k1 = jnp.ones((prefix, N_HEADS, D_HEAD), dtype=jnp.float32)
    k2 = -jnp.ones((prefix, N_HEADS, D_HEAD), dtype=jnp.float32)
    v = jnp.arange(prefix * N_HEADS * D_HEAD, dtype=jnp.float32).reshape(prefix, N_HEADS, D_HEAD)
    v = v / jnp.linalg.norm(v)
    logits1 = drafter_forward(params, h, k1, v, prefix_len=prefix, gamma=4)
    logits2 = drafter_forward(params, h, k2, v, prefix_len=prefix, gamma=4)
    assert logits1.shape == (4, VOCAB)
    assert not jnp.allclose(logits1, logits2, atol=1e-5, rtol=1e-5)


def test_rope_positions_continue_from_prefix():
    vec = jnp.arange(D_HEAD, dtype=jnp.float32) + 0.3
    y0 = apply_rope(vec[None, :], jnp.array([0]))
    y4 = apply_rope(vec[None, :], jnp.array([4]))
    assert not jnp.allclose(y0, y4)

    params = init_drafter(jax.random.PRNGKey(5))
    q0, _, _ = draft_qkv(params, gamma=4, prefix_len=0)
    q4, _, _ = draft_qkv(params, gamma=4, prefix_len=4)
    assert not jnp.allclose(q0[0], q4[0])
    pos = block_positions(4, 4)
    assert int(pos[0]) == 4
    assert int(block_positions(0, 4)[0]) == 0
    raw = jnp.einsum("gd,dhe->ghe", mask_inputs(params, 4), params["W_q"])
    at_zero = apply_rope(raw, block_positions(0, 4))
    at_prefix = apply_rope(raw, block_positions(4, 4))
    assert jnp.allclose(q4[0], at_prefix[0])
    assert not jnp.allclose(q4[0], at_zero[0])


def test_accepted_length_and_prefix_match_target():
    target_params = init_target(jax.random.PRNGKey(6))
    prefix = jnp.array([1, 4, 2, 7, 3], dtype=jnp.int32)
    gamma = 4
    target_tokens = target_greedy_tokens(target_params, prefix, gamma)
    assert target_tokens.shape == (gamma,)
    draft = target_tokens.at[2:].set((target_tokens[2:] + 1) % VOCAB)
    length = acceptance_length(draft, target_tokens)
    accepted = accepted_prefix(draft, target_tokens)
    assert 0 <= int(length) <= gamma
    assert int(length) == 2
    assert jnp.array_equal(accepted, target_tokens[:2])
    assert jnp.array_equal(accepted, draft[:2])

    full = acceptance_length(target_tokens, target_tokens)
    assert int(full) == gamma
    assert jnp.array_equal(accepted_prefix(target_tokens, target_tokens), target_tokens)


def test_first_reject_truncation():
    target = jnp.array([5, 6, 7, 8, 9], dtype=jnp.int32)
    gamma = target.shape[0]
    length0, = _as_list(acceptance_length(jnp.array([0, 6, 7, 8, 9]), target))
    assert length0 == 0
    assert accepted_prefix(jnp.array([0, 6, 7, 8, 9]), target).shape[0] == 0
    for j in range(gamma):
        draft = target.at[j].set((target[j] + 3) % VOCAB)
        length = int(acceptance_length(draft, target))
        assert length == j
        accepted = accepted_prefix(draft, target)
        assert jnp.array_equal(accepted, target[:j])
        if j < gamma:
            assert accepted.shape[0] == j
    assert int(acceptance_length(target, target)) == gamma


def _as_list(length):
    return (int(length),)


@pytest.mark.parametrize("prefix_len", [0, 4])
def test_block_position_zero_tracks_prefix(prefix_len):
    assert int(block_positions(prefix_len, 1)[0]) == prefix_len
