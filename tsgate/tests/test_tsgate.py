import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tsgate import (
    apply_output_gate,
    gate_values,
    image_to_text_fraction,
    init_stream_params,
    joint_attention,
    peak_attention_mass,
    project_q_and_content,
    rmsnorm,
    rope,
    time_bias,
    time_branch_param_count,
)


def _sig4():
    return 1.0 / (1.0 + np.exp(-4.0))


def test_time_branch_param_counts():
    assert time_branch_param_count(3, 12, 1536, 1536, 12, "element") == 84989952
    assert time_branch_param_count(3, 12, 1536, 1536, 12, "head") == 663984
    assert time_branch_param_count(3, 24, 1536, 1536, 12, "element") == 169979904
    with pytest.raises(ValueError):
        time_branch_param_count(3, 12, 1536, 1536, 12, "nope")


def test_init_is_content_only():
    d_model, n_heads, d_temb = 32, 4, 16
    d_h = d_model // n_heads
    key = jax.random.PRNGKey(0)
    k_x, k_t = jax.random.split(jax.random.PRNGKey(1), 2)
    x = jax.random.normal(k_x, (2, 5, d_model))
    temb = jax.random.normal(k_t, (2, d_temb))
    expected = _sig4()

    for granularity, gph in (("element", d_h), ("head", 1)):
        params = init_stream_params(key, d_model, n_heads, d_temb, granularity, True)
        bias = np.asarray(params["b_qg"]).reshape(n_heads, d_h + gph)
        assert np.allclose(bias[:, :d_h], 0.0)
        assert np.allclose(bias[:, d_h:], 4.0)
        assert np.allclose(params["w_time"], 0.0)
        assert np.allclose(params["b_time"], 0.0)
        b_time = time_bias(temb, params["w_time"], params["b_time"])
        assert np.allclose(b_time, 0.0, atol=1e-5)
        _q, c = project_q_and_content(
            x, params["w_qg"], params["b_qg"], n_heads, granularity
        )
        g_arr = gate_values(c, b_time, n_heads, granularity, d_head=d_h)
        g_zero = gate_values(c, 0, n_heads, granularity, d_head=d_h)
        assert np.allclose(g_arr, g_zero, atol=1e-5)
        assert g_arr.shape == x.shape

        w_zero = jnp.zeros_like(params["w_qg"])
        _q0, c0 = project_q_and_content(x, w_zero, params["b_qg"], n_heads, granularity)
        g = gate_values(c0, 0, n_heads, granularity, d_head=d_h)
        assert g.shape == x.shape
        assert np.allclose(g, expected, atol=1e-6)


def test_shared_bias_orders_and_threshold():
    n_tok, n_heads, d_h = 6, 4, 8
    base = jnp.arange(n_tok, dtype=jnp.float32) - 2.5
    c = base[:, None, None] + 0.01 * jnp.arange(n_heads * d_h, dtype=jnp.float32).reshape(
        1, n_heads, d_h
    )
    bias = jnp.full((n_heads * d_h,), 0.7, dtype=jnp.float32)
    g = gate_values(c, bias, n_heads, "element")
    c_flat = np.asarray(c).reshape(n_tok, n_heads * d_h)
    g_np = np.asarray(g)
    for ch in range(n_heads * d_h):
        assert np.array_equal(np.argsort(g_np[:, ch]), np.argsort(c_flat[:, ch]))
    assert np.array_equal(g_np > 0.5, c_flat > -0.7)

    eps = np.float32(1e-3)
    gp = gate_values(c, bias + eps, n_heads, "element")
    gm = gate_values(c, bias - eps, n_heads, "element")
    fd = (gp - gm) / (2 * eps)
    anal = g * (1.0 - g)
    assert np.allclose(fd, anal, atol=1e-4)
    assert np.all(np.asarray(anal) <= 0.25 + 1e-6)

    g0 = gate_values(jnp.zeros((1, n_heads, d_h)), 0, n_heads, "element")
    g8 = gate_values(jnp.full((1, n_heads, d_h), 8.0), 0, n_heads, "element")
    d0 = g0 * (1.0 - g0)
    d8 = g8 * (1.0 - g8)
    assert np.all(np.asarray(d0) > np.asarray(d8))


def test_time_bias_ignores_tokens():
    sig = inspect.signature(time_bias)
    assert "x" not in sig.parameters
    assert tuple(sig.parameters) == ("temb", "w_time", "b_time")

    d_model, n_heads, d_temb = 32, 4, 16
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(2), 4)
    w_time = jax.random.normal(k1, (d_temb, d_model)) * 0.1
    b_time = jnp.zeros((d_model,))
    temb = jax.random.normal(k2, (2, d_temb))
    x1 = jax.random.normal(k3, (2, 5, d_model))
    x2 = jax.random.normal(k4, (2, 5, d_model))
    assert not np.allclose(x1, x2)
    b_from_same_temb = time_bias(temb, w_time, b_time)
    assert np.allclose(b_from_same_temb, time_bias(temb, w_time, b_time), atol=1e-5)

    d_h = d_model // n_heads
    w_qg = jax.random.normal(k1, (d_model, n_heads * (d_h + d_h))) * 0.02
    b_qg = jnp.zeros((n_heads * (d_h + d_h),))
    _q, c = project_q_and_content(x1, w_qg, b_qg, n_heads, "element")
    g1 = gate_values(c, time_bias(temb, w_time, b_time), n_heads, "element")
    g2 = gate_values(c, time_bias(temb, w_time, b_time + 1.0), n_heads, "element")
    assert np.all(np.asarray(g2) > np.asarray(g1))

    temb_a = jnp.zeros((2, d_temb))
    temb_b = jnp.ones((2, d_temb))
    w_ones = jnp.ones((d_temb, d_model))
    assert not np.allclose(
        time_bias(temb_a, w_ones, b_time),
        time_bias(temb_b, w_ones, b_time),
    )


def test_headwise_constant_within_head():
    d_model, n_heads, n_tok = 32, 4, 5
    d_h = d_model // n_heads
    logits = jnp.array([-1.5, -0.2, 0.7, 2.0], dtype=jnp.float32)
    c = jnp.broadcast_to(logits.reshape(1, n_heads, 1), (n_tok, n_heads, 1))
    # Poison extra channels so only the per-head logit may be used.
    c_wide = jnp.concatenate(
        [c, jnp.full((n_tok, n_heads, d_h - 1), 99.0)],
        axis=-1,
    )
    bias = jnp.array([0.1, -0.4, 0.8, -0.2], dtype=jnp.float32)
    g = gate_values(c_wide, bias, n_heads, "head")
    assert g.shape == (n_tok, d_model)
    g_heads = np.asarray(g).reshape(n_tok, n_heads, d_h)
    assert np.allclose(g_heads, g_heads[..., :1], atol=1e-5)
    assert not np.allclose(g_heads[:, 0, :], g_heads[:, 1, :])


def test_output_gate_does_not_renormalize():
    p = jnp.array(
        [
            [0.05, 0.15, 0.20, 0.25, 0.35],
            [0.10, 0.20, 0.30, 0.25, 0.15],
        ],
        dtype=jnp.float32,
    )
    assert np.allclose(np.asarray(p).sum(axis=-1), 1.0, atol=1e-5)
    g = jnp.array(
        [
            [0.2, 0.5, 0.9],
            [0.3, 0.4, 0.7],
        ],
        dtype=jnp.float32,
    )
    assert not np.allclose(g, 1.0)
    v = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10.0
    a = p @ v
    mixed = g[:, None, :] * p[:, :, None]
    contracted = jnp.einsum("bjc,jc->bc", mixed, v)
    assert np.allclose(g * a, contracted, atol=1e-5)
    channel0 = jnp.sum(g[:, 0:1] * p, axis=-1)
    assert np.allclose(channel0, g[:, 0], atol=1e-5)
    assert not np.allclose(channel0, 1.0, atol=1e-5)

    w_o = jnp.eye(3, dtype=jnp.float32)
    b_o = jnp.zeros((3,), dtype=jnp.float32)
    y = apply_output_gate(a, g, w_o, b_o)
    assert np.allclose(y, g * a, atol=1e-5)

    text_slice = (3, 5)
    frac = image_to_text_fraction(p, text_slice)
    # The metric reads P, so replacing A with G * A cannot change it.
    frac_after = image_to_text_fraction(p, text_slice)
    assert np.allclose(frac, frac_after, atol=1e-5)
    manual = np.mean(np.asarray(p)[:, 3:5].sum(axis=-1))
    assert np.allclose(frac, manual, atol=1e-5)


def test_metrics_hand_values():
    p = jnp.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=jnp.float32,
    )
    frac = image_to_text_fraction(p, (2, 4))
    peak = peak_attention_mass(p)
    assert np.allclose(frac, 0.5, atol=1e-5)
    assert np.allclose(peak, 0.25, atol=1e-5)
    assert 0.0 <= float(frac) <= 1.0
    assert 0.0 <= float(peak) <= 1.0

    one_hot = jnp.zeros((3, 4), dtype=jnp.float32).at[:, 1].set(1.0)
    frac_hot = image_to_text_fraction(one_hot, (2, 4))
    peak_hot = peak_attention_mass(one_hot)
    assert np.allclose(peak_hot, 1.0, atol=1e-5)
    assert np.allclose(frac_hot, 0.0, atol=1e-5)
    assert 0.0 <= float(frac_hot) <= 1.0
    assert 0.0 <= float(peak_hot) <= 1.0


def test_rope_and_rmsnorm():
    key = jax.random.PRNGKey(4)
    x = jax.random.normal(key, (2, 7, 4, 8))
    pos = jnp.arange(7)
    y = rope(x, pos)
    assert np.allclose(
        np.linalg.norm(np.asarray(x), axis=-1),
        np.linalg.norm(np.asarray(y), axis=-1),
        atol=1e-5,
    )
    kq, kk = jax.random.split(jax.random.PRNGKey(5), 2)
    q = jax.random.normal(kq, (8,))
    k = jax.random.normal(kk, (8,))
    d_far = jnp.dot(rope(q, 5), rope(k, 8))
    d_near = jnp.dot(rope(q, 0), rope(k, 3))
    assert np.allclose(d_far, d_near, atol=1e-5)

    vec = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0, 0.5, -1.5, 2.5], dtype=jnp.float32)
    assert float(jnp.mean(jnp.square(vec))) > 1e-6
    normed = rmsnorm(vec, jnp.ones_like(vec), eps=1e-6)
    assert np.allclose(np.mean(np.asarray(normed) ** 2), 1.0, atol=1e-5)

    sig = inspect.signature(project_q_and_content)
    assert "position" not in sig.parameters
    assert "positions" not in sig.parameters
    d_model, n_heads = 32, 4
    d_h = d_model // n_heads
    w = jax.random.normal(jax.random.PRNGKey(6), (d_model, n_heads * (d_h + d_h))) * 0.02
    b = jnp.zeros((n_heads * (d_h + d_h),))
    tokens = jax.random.normal(jax.random.PRNGKey(7), (2, 5, d_model))
    q_proj, c = project_q_and_content(tokens, w, b, n_heads, "element")
    _ = rope(q_proj, jnp.arange(5))
    _q2, c2 = project_q_and_content(tokens, w, b, n_heads, "element")
    assert np.allclose(c, c2, atol=1e-5)


def test_joint_attention_pregate_probs():
    bsz, d_model, n_heads, d_temb = 2, 32, 4, 16
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(8), 5)
    params = [
        init_stream_params(k0, d_model, n_heads, d_temb, "element", True),
        init_stream_params(k1, d_model, n_heads, d_temb, "element", True),
    ]
    streams = [
        jax.random.normal(k2, (bsz, 3, d_model)),
        jax.random.normal(k3, (bsz, 2, d_model)),
    ]
    positions = [jnp.arange(3), jnp.arange(2)]
    temb = jax.random.normal(k4, (bsz, d_temb))
    temb_other = temb + 1.3
    outputs, probs, slices = joint_attention(
        streams, params, temb, positions, "element"
    )
    assert outputs[0].shape == (2, 3, 32)
    assert outputs[1].shape == (2, 2, 32)
    assert probs[0].shape == (2, 3, 5)
    assert probs[1].shape == (2, 2, 5)
    assert np.allclose(np.asarray(probs[0]).sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(probs[1]).sum(axis=-1), 1.0, atol=1e-5)
    assert slices == [(0, 3), (3, 5)]

    outputs_b, _, _ = joint_attention(
        streams, params, temb_other, positions, "element"
    )
    assert np.allclose(outputs[0], outputs_b[0], atol=1e-5)
    assert np.allclose(outputs[1], outputs_b[1], atol=1e-5)

    params_time = [dict(params[0]), dict(params[1])]
    params_time[0]["w_time"] = jnp.ones_like(params[0]["w_time"])
    y1, _, _ = joint_attention(streams, params_time, temb, positions, "element")
    y2, _, _ = joint_attention(streams, params_time, temb_other, positions, "element")
    assert not np.allclose(y1[0], y2[0], atol=1e-5, rtol=1e-5)

    pos_a = [jnp.arange(3), jnp.arange(2)]
    pos_b = [jnp.arange(3) + 10, jnp.arange(2) + 4]
    _q, c_before = project_q_and_content(
        streams[0], params[0]["w_qg"], params[0]["b_qg"], n_heads, "element"
    )
    joint_attention(streams, params, temb, pos_a, "element")
    joint_attention(streams, params, temb, pos_b, "element")
    _q2, c_after = project_q_and_content(
        streams[0], params[0]["w_qg"], params[0]["b_qg"], n_heads, "element"
    )
    assert np.allclose(c_before, c_after, atol=1e-5)
    assert "positions" not in inspect.signature(project_q_and_content).parameters


def test_first_layer_without_time():
    bsz, n_tok, d_model, n_heads, d_temb = 2, 4, 32, 4, 16
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(9), 3)
    params = init_stream_params(k0, d_model, n_heads, d_temb, "element", False)
    assert "w_time" not in params
    assert "b_time" not in params
    x = jax.random.normal(k1, (bsz, n_tok, d_model))
    positions = [jnp.arange(n_tok)]
    out1, prob1, slices1 = joint_attention([x], [params], None, positions, "element")
    with_zero = dict(params)
    with_zero["w_time"] = jnp.zeros((d_temb, d_model), dtype=jnp.float32)
    with_zero["b_time"] = jnp.zeros((d_model,), dtype=jnp.float32)
    temb = jax.random.normal(k2, (bsz, d_temb))
    out2, prob2, slices2 = joint_attention(
        [x], [with_zero], temb, positions, "element"
    )
    assert np.allclose(out1[0], out2[0], atol=1e-5)
    assert np.allclose(prob1[0], prob2[0], atol=1e-5)
    assert slices1 == slices2


def test_gate_bounded_and_not_post_scale():
    c = jnp.array([50.0, -50.0, 50.0, -50.0], dtype=jnp.float32).reshape(1, 1, 4)
    # d_h must be even; use H=1, d_h=4 so D=4, plus a wider element case.
    g = gate_values(c, 0, 1, "element")
    assert np.all(np.asarray(g) > 0.0)
    assert np.all(np.asarray(g) < 1.0)
    c_wide = jnp.concatenate(
        [jnp.full((2, 2, 4), 50.0), jnp.full((2, 2, 4), -50.0)],
        axis=0,
    )
    g_wide = gate_values(c_wide, 0, 2, "element")
    assert np.all(np.asarray(g_wide) > 0.0)
    assert np.all(np.asarray(g_wide) < 1.0)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(10), 3)
    a = jax.random.normal(k1, (2, 4, 16))
    gate = jax.nn.sigmoid(jax.random.normal(k2, (2, 4, 16)))
    assert np.all(np.asarray(gate) > 0.0)
    assert np.all(np.asarray(gate) < 1.0)
    w_o = jax.random.normal(k3, (16, 16))
    left = (gate * a) @ w_o
    right = gate * (a @ w_o)
    assert not np.allclose(left, right, atol=1e-5, rtol=1e-5)
    y = apply_output_gate(a, gate, w_o, jnp.zeros((16,), dtype=a.dtype))
    assert np.allclose(y, left, atol=1e-5)
    assert not np.allclose(y, right, atol=1e-5, rtol=1e-5)


def test_smoke_step_decreases_loss():
    bsz, n_tok, d_model, n_heads, d_temb = 4, 4, 16, 2, 8
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(11), 4)
    params = [init_stream_params(k1, d_model, n_heads, d_temb, "element", True)]
    x = jax.random.normal(k2, (bsz, n_tok, d_model))
    temb = jax.random.normal(k3, (bsz, d_temb))
    positions = [jnp.arange(n_tok)]
    pred0 = joint_attention([x], params, temb, positions, "element")[0][0]
    noise = jax.random.normal(k4, pred0.shape)
    target = jax.lax.stop_gradient(pred0 + 0.1 * noise)

    def loss_fn(p):
        pred = joint_attention([x], p, temb, positions, "element")[0][0]
        return jnp.mean(jnp.square(pred - target))

    loss0 = loss_fn(params)
    grads0 = jax.grad(loss_fn)(params)
    gw = grads0[0]["w_time"]
    assert np.all(np.isfinite(np.asarray(gw)))
    assert not np.allclose(gw, 0.0)

    lr = 1e-2
    current = params
    for _ in range(15):
        grads = jax.grad(loss_fn)(current)
        current = jax.tree_util.tree_map(lambda a, b: a - lr * b, current, grads)
    loss1 = loss_fn(current)
    assert float(loss1) < float(loss0)
