"""Shape tests for per-port AmpNet assembly and a tiny AdamW run."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_nmd.config import NMDConfig
from jax_nmd.train import (
    adamw_update,
    forward,
    init_adamw,
    init_params,
    make_synthetic_batch,
    predict_loss,
    train_step,
)


def _cfg(**kwargs):
    base = dict(k_poles=4, d_t=32, n_tokens=4, hidden=32, conv_mid=8)
    base.update(kwargs)
    return NMDConfig(**base)


def test_two_port_and_four_port_shapes():
    cfg = _cfg()
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    geom = jnp.ones((2, cfg.geom_h, cfg.geom_w))
    omega = jnp.linspace(1.0, 10.0, 5)
    ports2 = jnp.zeros((2, 2, cfg.port_dim))
    out2 = forward(params, geom, ports2, omega, cfg)
    assert out2["Z"].shape == (2, 5, 2, 2)
    assert out2["S"].shape == (2, 5, 2, 2)
    assert out2["poles"].shape == (2, cfg.k_poles)
    assert out2["left"].shape == (2, cfg.k_poles, 2)
    assert out2["right"].shape == (2, cfg.k_poles, 2)
    ports4 = jnp.zeros((2, 4, cfg.port_dim))
    out4 = forward(params, geom, ports4, omega, cfg)
    assert out4["Z"].shape == (2, 5, 4, 4)
    assert out4["S"].shape == (2, 5, 4, 4)
    assert out4["left"].shape == (2, cfg.k_poles, 4)
    np.testing.assert_allclose(
        np.asarray(out4["poles"]), np.asarray(out2["poles"]), atol=1e-5
    )


def test_three_port_from_three_queries():
    cfg = _cfg()
    key = jax.random.PRNGKey(1)
    params = init_params(key, cfg)
    geom = jax.random.bernoulli(key, 0.5, (1, cfg.geom_h, cfg.geom_w)).astype(jnp.float32)
    omega = jnp.linspace(1.0, 5.0, 6)
    ports3 = jnp.array([[[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]])
    out = forward(params, geom, ports3, omega, cfg)
    assert out["Z"].shape == (1, 6, 3, 3)
    assert out["S"].shape == (1, 6, 3, 3)
    assert out["left"].shape == (1, cfg.k_poles, 3)
    assert out["right"].shape == (1, cfg.k_poles, 3)


def test_tokens_shape():
    cfg = _cfg()
    params = init_params(jax.random.PRNGKey(2), cfg)
    geom = jnp.zeros((3, 8, 8))
    ports = jnp.zeros((3, 2, 2))
    omega = jnp.linspace(1.0, 2.0, 4)
    out = forward(params, geom, ports, omega, cfg)
    assert out["tokens"].shape == (3, cfg.n_tokens, cfg.d_t)


def test_adamw_few_steps_loss_decreases():
    cfg = _cfg(k_poles=3, hidden=32)
    key = jax.random.PRNGKey(3)
    k_init, k_data = jax.random.split(key)
    params = init_params(k_init, cfg)
    batch = make_synthetic_batch(k_data, cfg, batch_size=1, n_ports=2, n_freq=8)
    loss0, _ = predict_loss(params, batch, cfg)
    opt = init_adamw(params)
    p, o = params, opt
    for _ in range(12):
        p, o, _ = train_step(p, o, batch, cfg)
    loss_n, _ = predict_loss(p, batch, cfg)
    loss0 = float(loss0)
    loss_n = float(loss_n)
    assert np.isfinite(loss0) and np.isfinite(loss_n)
    assert loss_n < loss0


def test_adamw_update_changes_params():
    cfg = _cfg()
    params = init_params(jax.random.PRNGKey(4), cfg)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = init_adamw(params)
    new_p, _ = adamw_update(params, grads, opt, lr=cfg.lr, wd=cfg.weight_decay)
    same = jax.tree_util.tree_map(lambda a, b: bool(np.allclose(a, b)), params, new_p)
    assert not all(jax.tree_util.tree_leaves(same))
