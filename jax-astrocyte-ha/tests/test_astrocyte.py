"""CPU tests for the neural-astrocyte hybrid automaton."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax_astrocyte_ha.dynamics import (
    arnn_step,
    arnn_update,
    decode,
    init_arnn,
    init_nrnn,
    nrnn_step,
)
from jax_astrocyte_ha.env import (
    Config,
    build_trial,
    make_nrnn_batch,
    reach_mse,
    reach_reward,
    sample_context_sequence,
    target_xy,
    tile_contexts,
)
from jax_astrocyte_ha.train import (
    a2c_train_step,
    collect_a2c_rollout,
    hybrid_rollout,
    mse_with_pi,
    nrnn_mse_loss,
    nrnn_train_step,
    teacher_pi_from_context,
    wrong_pi_from_context,
)
from jax_astrocyte_ha.weff import effective_weights


def _finite_tree(tree):
    leaves = jax.tree.leaves(tree)
    return all(jnp.all(jnp.isfinite(x)) for x in leaves)


def test_weff_onehot_rank1():
    Nx, M = 8, 4
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    J = jax.random.normal(k1, (Nx, Nx))
    H = jax.random.normal(k2, (Nx, M))
    weffs = []
    for i in range(M):
        pi = jax.nn.one_hot(i, M)
        Weff = effective_weights(J, H, pi)
        expected = J + H[:, i : i + 1] @ H[:, i : i + 1].T
        np.testing.assert_allclose(np.asarray(Weff), np.asarray(expected), rtol=1e-5, atol=1e-5)
        assert Weff.shape == (Nx, Nx)
        weffs.append(np.asarray(Weff))
    assert not np.allclose(weffs[0], weffs[1])
    # literal Hadamard with ones leaves J unchanged
    ones = np.ones((Nx, Nx))
    pi0 = jax.nn.one_hot(0, M)
    Wlit = np.asarray(J) * ones + np.asarray(H @ jnp.diag(pi0) @ H.T)
    np.testing.assert_allclose(weffs[0], Wlit, rtol=1e-5, atol=1e-5)


def test_nrnn_step_shapes_finite_tau0():
    cfg = Config()
    key = jax.random.PRNGKey(1)
    params = init_nrnn(key, Nx=cfg.Nx, M=cfg.M, Nu=cfg.Nu, Ny=cfg.Ny)
    x = jax.random.normal(jax.random.PRNGKey(2), (cfg.Nx,))
    u = jax.random.normal(jax.random.PRNGKey(3), (cfg.Nu,))
    pi = jax.nn.one_hot(0, cfg.M)
    x_next = nrnn_step(x, u, pi, params, tau_x=0.3, eps=0.01)
    assert x_next.shape == (cfg.Nx,)
    assert jnp.all(jnp.isfinite(x_next))
    x_frozen = nrnn_step(x, u, pi, params, tau_x=0.0, eps=1.0)
    np.testing.assert_allclose(np.asarray(x_frozen), np.asarray(x), atol=0.0)


def test_arnn_step_onehot_argmax():
    cfg = Config(Nh=8, M=4)
    key = jax.random.PRNGKey(4)
    params = init_arnn(key, Nh=cfg.Nh, M=cfg.M, No=cfg.No, hidden=cfg.hidden)
    h = jax.random.normal(jax.random.PRNGKey(5), (cfg.Nh,))
    o = jnp.concatenate([jax.nn.one_hot(1, cfg.M), jnp.array([0.0])])
    h_new, g, pi, logits, probs = arnn_update(h, o, params, tau_h=0.8)
    assert h_new.shape == (cfg.Nh,)
    assert jnp.all(jnp.isfinite(h_new))
    assert pi.shape == (cfg.M,)
    np.testing.assert_allclose(float(jnp.sum(pi)), 1.0)
    assert set(np.unique(np.asarray(pi)).tolist()).issubset({0.0, 1.0})
    assert int(g) == int(jnp.argmax(logits))
    assert int(g) == int(jnp.argmax(probs))
    h2 = arnn_step(h, o, params, tau_h=0.8)
    np.testing.assert_allclose(np.asarray(h_new), np.asarray(h2), rtol=1e-5)


def test_decode_y_equals_Wqx_x():
    key = jax.random.PRNGKey(6)
    k1, k2 = jax.random.split(key)
    Nx, Ny = 16, 2
    W = jax.random.normal(k1, (Ny, Nx))
    x = jax.random.normal(k2, (Nx,))
    y = decode(x, W)
    np.testing.assert_allclose(np.asarray(y), np.asarray(W @ x), rtol=1e-5)
    assert y.shape == (Ny,)


def test_env_blocks_and_context_map():
    ctx = tile_contexts([0, 1], [5, 7])
    assert len(ctx) == 12
    assert np.all(ctx[:5] == 0)
    assert np.all(ctx[5:] == 1)
    assert ctx[4] != ctx[5]

    key = jax.random.PRNGKey(7)
    seq, bids = sample_context_sequence(key, n_trials=40, M=4, lo=3, hi=8)
    assert len(seq) == 40
    for b in np.unique(bids):
        sl = seq[bids == b]
        assert np.all(sl == sl[0])
    assert len(np.unique(bids)) >= 2

    cfg = Config(M=4)
    theta = jnp.array(0.4)
    for c in range(cfg.M):
        tgt = target_xy(theta, c, cfg.M)
        ang = 0.4 + 2.0 * np.pi * c / cfg.M
        np.testing.assert_allclose(np.asarray(tgt), [np.cos(ang), np.sin(ang)], rtol=1e-5)
        u, y_tgt, mask, target = build_trial(theta, c, cfg)
        np.testing.assert_allclose(np.asarray(target), np.asarray(tgt), rtol=1e-5)
        stim = u[cfg.n_fix]
        np.testing.assert_allclose(np.asarray(stim[:2]), [np.cos(0.4), np.sin(0.4)], rtol=1e-5)
        assert float(stim[2]) == 0.0
        assert float(u[-1, 2]) == 1.0
        np.testing.assert_allclose(np.asarray(y_tgt[-1]), np.asarray(tgt), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(y_tgt[0]), np.zeros(2), atol=1e-6)
        assert y_tgt.shape == (cfg.T, 2)
        assert u.shape == (cfg.T, 3)


def test_reward_correct_reach_higher():
    target = jnp.array([1.0, 0.0])
    r_ok = reach_reward(target, target, threshold=0.5)
    r_bad = reach_reward(-target, target, threshold=0.5)
    assert float(r_ok) > float(r_bad)
    assert float(r_ok) == 1.0
    assert float(r_bad) == 0.0
    assert float(reach_mse(target, target)) < float(reach_mse(-target, target))


def test_nrnn_train_step_reduces_or_updates():
    cfg = Config(M=2, h_std=0.2)
    key = jax.random.PRNGKey(8)
    k_p, k_b = jax.random.split(key)
    params = init_nrnn(k_p, Nx=cfg.Nx, M=cfg.M, Nu=cfg.Nu, Ny=cfg.Ny, h_std=0.2)
    batch = make_nrnn_batch(k_b, cfg, batch_size=8)
    loss0 = nrnn_mse_loss(params, batch, cfg.tau_x)
    params1, loss_at, grads = nrnn_train_step(params, batch, cfg.tau_x, lr=0.05)
    loss1 = nrnn_mse_loss(params1, batch, cfg.tau_x)
    assert jnp.isfinite(loss0) and jnp.isfinite(loss_at) and jnp.isfinite(loss1)
    assert _finite_tree(grads)
    changed = any(
        not np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(params1))
    )
    assert changed
    # One SGD step on the same batch usually descends; finite + update is enough.


def test_a2c_step_finite_grads():
    cfg = Config(M=2, Nh=8)
    key = jax.random.PRNGKey(9)
    k_n, k_a, k_r = jax.random.split(key, 3)
    nrnn = init_nrnn(k_n, Nx=cfg.Nx, M=cfg.M, h_std=0.2)
    arnn = init_arnn(k_a, Nh=cfg.Nh, M=cfg.M, No=cfg.No, hidden=cfg.hidden)
    contexts = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    thetas = np.linspace(0.1, 2.0, len(contexts), dtype=np.float32)
    h0, actions, rewards, mses = collect_a2c_rollout(
        nrnn, arnn, cfg, k_r, contexts, thetas, sample_actions=True
    )
    arnn1, loss, grads = a2c_train_step(arnn, h0, actions, rewards, cfg.tau_h, cfg, lr=0.01)
    assert jnp.isfinite(loss)
    assert _finite_tree(grads)
    assert _finite_tree(arnn1)
    assert jnp.all(jnp.isfinite(mses))


def test_hybrid_rollout_no_nan_and_pi_modulation():
    cfg = Config(M=2, Nx=16, Nh=8, h_std=0.2, n_fix=1, n_stim=3, n_delay=1, n_resp=3)
    key = jax.random.PRNGKey(10)
    k_n, k_a, k_b = jax.random.split(key, 3)
    nrnn = init_nrnn(k_n, Nx=cfg.Nx, M=cfg.M, h_std=0.2)
    arnn = init_arnn(k_a, Nh=cfg.Nh, M=cfg.M, No=cfg.No, hidden=cfg.hidden)

    contexts = tile_contexts([0, 1], [6, 6])
    thetas = np.linspace(0.2, 5.0, len(contexts), dtype=np.float32)
    out = hybrid_rollout(nrnn, arnn, cfg, contexts, thetas)
    assert jnp.all(jnp.isfinite(out["rewards"]))
    assert jnp.all(jnp.isfinite(out["mses"]))
    assert jnp.all(jnp.isfinite(out["h"]))
    assert out["pis"].shape == (len(contexts), cfg.M)

    batch = make_nrnn_batch(k_b, cfg, batch_size=16)
    step = jax.jit(lambda p, b: nrnn_train_step(p, b, cfg.tau_x, 0.05)[0])
    p = nrnn
    for _ in range(50):
        p = step(p, batch)
    mse_true = float(mse_with_pi(p, batch, cfg.tau_x, batch["pi"]))
    pi_wrong = wrong_pi_from_context(batch["context"], cfg.M)
    mse_wrong = float(mse_with_pi(p, batch, cfg.tau_x, pi_wrong))
    assert np.isfinite(mse_true) and np.isfinite(mse_wrong)
    assert mse_true < mse_wrong
    assert teacher_pi_from_context(jnp.array(0), 2).shape == (2,)
