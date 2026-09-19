"""Equation-level tests for the DRFC / DRFC-Seq kernel. Fail if the logic is wrong."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_drfc.certificates import (
    anytime_radius,
    beta_radius,
    eliminate_once,
    equal_agent_means,
    pairwise_ghat_and_B,
)
from jax_drfc.drfc import DRFCConfig, run_drfc
from jax_drfc.drfc_seq import DRFCSeqConfig, run_drfc_seq, seq_switch_decision
from jax_drfc.env import (
    aggregate_block,
    balanced_block_schedule,
    count_s_loc_s_dec,
    make_cancellation_env,
)


def test_balanced_block_each_active_arm_h_times():
    key = jax.random.PRNGKey(0)
    n_agents, h = 3, 5
    active = jnp.array([0, 2, 4], dtype=jnp.int32)
    schedule = balanced_block_schedule(key, n_agents, active, h)
    assert schedule.shape == (n_agents, int(active.shape[0]) * h)
    for i in range(n_agents):
        row = np.asarray(schedule[i])
        for a in np.asarray(active):
            assert int(np.sum(row == int(a))) == h
        for a in (1, 3):
            assert int(np.sum(row == a)) == 0

    env = make_cancellation_env(horizon=40, n_agents=n_agents, n_arms=5, noise="none")
    rewards, new_t = env.step_block(0, schedule, jax.random.PRNGKey(1))
    assert new_t == int(active.shape[0]) * h
    _means, counts = aggregate_block(schedule, rewards, n_arms=5)
    for i in range(n_agents):
        for a in np.asarray(active):
            assert int(np.asarray(counts[i, int(a)])) == h
        for a in (1, 3):
            assert int(np.asarray(counts[i, a])) == 0


def test_equal_agent_gap_cancels_across_agents():
    # Agent 0 has means (0.9, 0.1), agent 1 has (0.1, 0.9). Local gaps are large
    # and opposite; the equal-agent gap Gbar_{0,1} is 0.
    block_means = jnp.array(
        [[[0.9, 0.1], [0.1, 0.9]]],
        dtype=jnp.float32,
    )
    block_active = jnp.array([[True, True]])
    ghat, n_blocks = pairwise_ghat_and_B(block_means, block_active)
    np.testing.assert_allclose(np.asarray(ghat[0, 1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ghat[1, 0]), 0.0, atol=1e-6)
    assert int(n_blocks[0, 1]) == 1
    local0 = float(block_means[0, 0, 0] - block_means[0, 0, 1])
    local1 = float(block_means[0, 1, 0] - block_means[0, 1, 1])
    assert abs(local0 - 0.8) < 1e-6
    assert abs(local1 + 0.8) < 1e-6
    eq = equal_agent_means(block_means[0])
    np.testing.assert_allclose(np.asarray(eq), np.array([0.5, 0.5]), atol=1e-6)


def test_elimination_when_gap_beats_radius():
    block_means = jnp.array(
        [[[0.9, 0.1], [0.9, 0.1]]],
        dtype=jnp.float32,
    )
    block_active = jnp.array([[True, True]])
    ghat, _n_blocks = pairwise_ghat_and_B(block_means, block_active)
    np.testing.assert_allclose(np.asarray(ghat[0, 1]), 0.8, atol=1e-5)
    beta = jnp.full((2, 2), 0.01, dtype=jnp.float32)
    active = jnp.array([True, True])
    new_active = eliminate_once(active, ghat, beta)
    assert bool(new_active[0]) and (not bool(new_active[1]))

    beta_large = jnp.full((2, 2), 0.9, dtype=jnp.float32)
    still = eliminate_once(active, ghat, beta_large)
    assert bool(still[0]) and bool(still[1])


def test_drfc_no_switch_on_local_only_cancellation():
    horizon, period = 80, 16
    env = make_cancellation_env(
        horizon=horizon,
        n_agents=2,
        n_arms=2,
        delta=0.25,
        eps=0.20,
        period=period,
        switch_time=None,
        noise="none",
    )
    s_loc, s_dec = count_s_loc_s_dec(env.mean_path)
    assert s_dec == 0
    assert s_loc >= 3

    mu_flip = np.asarray(env.mean_path[period])
    assert mu_flip[0, 1] > mu_flip[0, 0]
    assert mu_flip.mean(axis=0)[0] > mu_flip.mean(axis=0)[1]

    cfg = DRFCConfig(
        n_agents=2,
        n_arms=2,
        horizon=horizon,
        h=2,
        M=20,
        Hscan=60,
        delta=0.1,
        C=0.15,
        init_candidate=0,
    )
    result = run_drfc(env, cfg, jax.random.PRNGKey(1))
    assert result.n_switches == 0
    assert result.candidate == 0
    assert result.n_certified >= 1


def test_drfc_seq_window_not_full_no_switch():
    muhat = jnp.array([0.1, 0.9], dtype=jnp.float32)
    cfg = DRFCSeqConfig(
        n_agents=2, n_arms=2, horizon=30, W=4, delta_p=100, h=2, alpha=0.05
    )
    env = make_cancellation_env(horizon=30, n_agents=2, n_arms=2, noise="none")
    result = run_drfc_seq(env, cfg, jax.random.PRNGKey(0))
    assert result.window_len < cfg.W
    assert result.n_switches == 0

    new_c, switched = seq_switch_decision(
        muhat, candidate=0, n=40, s=1, n_agents=2, n_arms=2, alpha=0.05
    )
    assert switched
    assert new_c == 1


def test_drfc_seq_full_window_positive_lc_gap_switches():
    muhat = jnp.array([0.05, 0.95], dtype=jnp.float32)
    n, s = 40, 1
    r = float(anytime_radius(n, s, n_agents=2, n_arms=2, alpha=0.05))
    assert (0.95 - 0.05) - (r + r) > 0
    new_c, switched = seq_switch_decision(
        muhat, candidate=0, n=n, s=s, n_agents=2, n_arms=2, alpha=0.05
    )
    assert switched
    assert new_c == 1

    close = jnp.array([0.50, 0.51], dtype=jnp.float32)
    _c2, switched2 = seq_switch_decision(
        close, candidate=0, n=n, s=s, n_agents=2, n_arms=2, alpha=0.05
    )
    assert not switched2


def test_radius_decreases_with_N_h_B():
    kwargs = dict(n_arms=3, horizon=100, delta=0.05, C=8.0)
    b = float(beta_radius(n_agents=2, h=4, n_blocks=3, **kwargs))
    b_N = float(beta_radius(n_agents=8, h=4, n_blocks=3, **kwargs))
    b_h = float(beta_radius(n_agents=2, h=16, n_blocks=3, **kwargs))
    b_B = float(beta_radius(n_agents=2, h=4, n_blocks=12, **kwargs))
    assert b_N < b
    assert b_h < b
    assert b_B < b
    assert b > 0
