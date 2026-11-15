"""CPU tests that check FlowATC paper equations, not just that code runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_flowatc.baselines import constant_velocity
from jax_flowatc.cfm import (
    euler_integrate,
    euler_path,
    interpolant,
    sample_logit_normal,
    target_velocity,
)
from jax_flowatc.config import paper_tiny_config, tiny_config
from jax_flowatc.data import deltas_and_elapsed, synthetic_straight, zscore_stats, apply_zscore
from jax_flowatc.layers import adaln, adaln_from_stats, block_causal_mask
from jax_flowatc.metrics import min_ade_at_k, min_fde_at_k
from jax_flowatc.model import (
    apply_one_block,
    block_attention_weights,
    eq11_concat,
    init_params,
    velocity_field,
)
from jax_flowatc.train import adamw_init, make_train_step


def test_adaln_eq9():
    h = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.0, -2.0, 4.0, 6.0]], dtype=jnp.float32)
    mu = jnp.mean(h, axis=-1, keepdims=True)
    sigma = jnp.sqrt(jnp.mean((h - mu) ** 2, axis=-1, keepdims=True))
    gamma = jnp.array([1.0, 0.5, 2.0, -1.0], dtype=jnp.float32)
    beta = jnp.array([0.1, -0.2, 0.3, 0.0], dtype=jnp.float32)
    expected = gamma * (h - mu) / sigma + beta
    got = adaln_from_stats(h, mu, sigma, gamma, beta)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-6, atol=1e-6)
    got_full = adaln(h, gamma, beta, eps=0.0)
    np.testing.assert_allclose(np.asarray(got_full), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_block_causal_mask_and_independence():
    cfg = tiny_config()
    t_obs, t_fut = cfg.t_obs, cfg.t_fut
    t_len = t_obs + t_fut
    mask = block_causal_mask(t_len, t_obs)
    assert mask.shape == (t_len, t_len)
    for i in range(t_len):
        for j in range(t_len):
            if i < t_obs and j >= t_obs:
                assert jnp.isneginf(mask[i, j])
            else:
                assert mask[i, j] == 0.0

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    k1, k2, k3 = jax.random.split(key, 3)
    x_obs = jax.random.normal(k1, (t_obs, cfg.state_dim))
    x_fut = jax.random.normal(k2, (t_fut, cfg.state_dim))
    x_fut_alt = x_fut + 3.7
    x_obs_alt = x_obs + 2.1
    times = jnp.cumsum(jnp.array([0.0, 1.2, 0.8, 2.0, 1.5, 0.6, 2.2, 1.1]))
    dt, t_rel = deltas_and_elapsed(times)
    t = jnp.array(0.4, dtype=jnp.float32)

    w = block_attention_weights(params, cfg, x_obs, x_fut, t, dt, t_rel, layer=0)
    # observed queries must put exactly 0 mass on future keys
    np.testing.assert_allclose(
        np.asarray(w[:, :t_obs, t_obs:]), 0.0, atol=0.0, rtol=0.0
    )
    # rows still sum to 1
    np.testing.assert_allclose(np.asarray(w.sum(axis=-1)), 1.0, rtol=1e-5, atol=1e-5)

    h = apply_one_block(params, cfg, x_obs, x_fut, t, dt, t_rel)
    h_fut_changed = apply_one_block(params, cfg, x_obs, x_fut_alt, t, dt, t_rel)
    np.testing.assert_allclose(
        np.asarray(h[:t_obs]), np.asarray(h_fut_changed[:t_obs]), rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(np.asarray(h[t_obs:]), np.asarray(h_fut_changed[t_obs:]), atol=1e-4)

    h_obs_changed = apply_one_block(params, cfg, x_obs_alt, x_fut, t, dt, t_rel)
    assert not np.allclose(np.asarray(h[t_obs:]), np.asarray(h_obs_changed[t_obs:]), atol=1e-4)


def test_eq11_concat_shape():
    cfg = tiny_config()
    key = jax.random.PRNGKey(1)
    params = init_params(key, cfg)
    x_obs = jnp.ones((cfg.t_obs, cfg.state_dim))
    x_tilde = jnp.zeros((cfg.t_fut, cfg.state_dim))
    z = eq11_concat(params.w_in, x_obs, x_tilde)
    assert z.shape == (cfg.t_obs + cfg.t_fut, cfg.d_model)
    np.testing.assert_allclose(
        np.asarray(z[: cfg.t_obs]), np.asarray(x_obs @ params.w_in), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(z[cfg.t_obs :]), np.asarray(x_tilde @ params.w_in), rtol=1e-6
    )


def test_cfm_interpolant_hand_example():
    x0 = jnp.array([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    x_fut = jnp.array([[2.0, 2.0], [3.0, 3.0]], dtype=jnp.float32)
    t = jnp.array(0.25, dtype=jnp.float32)
    x_t = interpolant(x0, x_fut, t)
    expected_xt = jnp.array([[0.5, 0.5], [1.5, 1.5]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(x_t), np.asarray(expected_xt), rtol=1e-6)
    vel = target_velocity(x0, x_fut)
    expected_vel = jnp.array([[2.0, 2.0], [2.0, 2.0]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(vel), np.asarray(expected_vel), rtol=1e-6)


def test_logit_normal_in_unit_interval():
    key = jax.random.PRNGKey(2)
    t = sample_logit_normal(key, (4096,), mean=1.0, std=1.0)
    assert t.shape == (4096,)
    assert jnp.all(t > 0.0) and jnp.all(t < 1.0)
    # mass should sit above 0.5 because logit mean is 1.0
    assert float(jnp.mean(t)) > 0.5


def test_euler_constant_velocity_field():
    x0 = jnp.array([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)
    x_fut = jnp.array([[4.0, 2.0], [5.0, 3.0]], dtype=jnp.float32)
    u = x_fut - x0

    def u_fn(x, t):
        return u

    one = euler_integrate(u_fn, x0, n_steps=1)
    np.testing.assert_allclose(np.asarray(one), np.asarray(x_fut), rtol=1e-5, atol=1e-5)

    path = euler_path(u_fn, x0, n_steps=20)
    assert path.shape == (20, 2, 2)
    for k in (0, 4, 9, 19):
        tt = (k + 1) / 20.0
        expected = (1.0 - tt) * x0 + tt * x_fut
        np.testing.assert_allclose(np.asarray(path[k]), np.asarray(expected), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(path[-1]), np.asarray(x_fut), rtol=1e-5, atol=1e-5)


def test_min_ade_fde_hand_example():
    # T_fut = 2, K = 2, features (x, y)
    gt = jnp.array([[0.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)
    s0 = jnp.array([[1.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)  # ADE=0.5, FDE=0
    s1 = jnp.array([[0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)  # ADE=2.5, FDE=5
    preds = jnp.stack([s0, s1], axis=0)
    ade = min_ade_at_k(preds, gt)
    fde = min_fde_at_k(preds, gt)
    np.testing.assert_allclose(float(ade), 0.5, rtol=1e-6)
    np.testing.assert_allclose(float(fde), 0.0, atol=1e-6)


def test_constant_velocity_exact():
    # last obs at t=10, p=(0,0), v=(2,1); future at t=11 and t=13
    x_obs = jnp.zeros((3, 6), dtype=jnp.float32)
    x_obs = x_obs.at[-1, 3].set(2.0)
    x_obs = x_obs.at[-1, 4].set(1.0)
    times_obs = jnp.array([8.0, 9.0, 10.0], dtype=jnp.float32)
    times_fut = jnp.array([11.0, 13.0], dtype=jnp.float32)
    pred = constant_velocity(x_obs, times_obs, times_fut)
    expected = jnp.array([[2.0, 1.0], [6.0, 3.0]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(pred), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_jitted_train_step_loss_decreases():
    cfg = tiny_config(dropout=0.0)
    key = jax.random.PRNGKey(7)
    k_data, k_init, k_train = jax.random.split(key, 3)
    states, times = synthetic_straight(k_data, batch=8, t_obs=cfg.t_obs, t_fut=cfg.t_fut)
    stats = zscore_stats(states)
    states = apply_zscore(states, stats)
    dt, t_rel = deltas_and_elapsed(times)
    x_obs = states[:, : cfg.t_obs]
    x_fut = states[:, cfg.t_obs :]
    params = init_params(k_init, cfg)
    opt = adamw_init(params)
    step = make_train_step(cfg, lr=3e-3, wd=0.0)

    params, opt, loss0, grads = step(params, opt, x_obs, x_fut, dt, t_rel, k_train)
    assert jnp.isfinite(loss0)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)

    keys = jax.random.split(k_train, 31)
    loss = loss0
    for i in range(1, 30):
        params, opt, loss, grads = step(params, opt, x_obs, x_fut, dt, t_rel, keys[i])
    assert jnp.isfinite(loss)
    assert float(loss) < float(loss0)


def test_forward_deterministic_dropout_off():
    cfg = tiny_config(dropout=0.0)
    key = jax.random.PRNGKey(3)
    params = init_params(key, cfg)
    k1, k2 = jax.random.split(key)
    x_obs = jax.random.normal(k1, (cfg.t_obs, cfg.state_dim))
    x_noisy = jax.random.normal(k2, (cfg.t_fut, cfg.state_dim))
    times = jnp.linspace(0.0, 10.0, cfg.seq_len)
    dt, t_rel = deltas_and_elapsed(times)
    t = jnp.array(0.31, dtype=jnp.float32)
    y1 = velocity_field(params, cfg, x_obs, x_noisy, t, dt, t_rel, None)
    y2 = velocity_field(params, cfg, x_obs, x_noisy, t, dt, t_rel, None)
    np.testing.assert_allclose(np.asarray(y1), np.asarray(y2), rtol=0.0, atol=0.0)
    assert y1.shape == (cfg.t_fut, cfg.state_dim)


def test_paper_tiny_config_not_run():
    cfg = paper_tiny_config()
    assert cfg.n_layers == 5
    assert cfg.n_heads == 4
    assert cfg.d_model == 128
    assert cfg.t_obs == 43 and cfg.t_fut == 43
