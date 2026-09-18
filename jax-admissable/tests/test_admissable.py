"""CPU tests for adversarial-missingness training (arXiv:2609.15297)."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jax_admissable.env import ToyPointEnv, readable_policy
from jax_admissable.missingness import MissingnessWrapper, mask_observation
from jax_admissable.pmiss import EPS, L_BOUND, init_pmiss, update_pmiss, update_pmiss_baseline
from jax_admissable.sac import SAC
from jax_admissable.train import estimate_adversarial_scores, make_readable_fn, train


def test_mask_zeros_feature_and_indicator():
    d = 6
    s = np.arange(d, dtype=np.float32) + 1.0
    u = 2
    out = mask_observation(s, u)
    assert out.shape == (2 * d,)
    masked, indicator = out[:d], out[d:]
    assert masked[u] == 0.0
    for i in range(d):
        if i != u:
            assert masked[i] == s[i]
    assert indicator[u] == 1.0
    assert np.sum(indicator) == 1.0
    np.testing.assert_array_equal(indicator, np.eye(d, dtype=np.float32)[u])


def test_k1_single_missing_index_constant_across_episode():
    env = ToyPointEnv(obs_dim=6, episode_length=12, seed=0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1.0
    wrap = MissingnessWrapper(env, p_miss=p, rng=np.random.default_rng(0))
    obs, u = wrap.reset()
    assert u == 4
    d = env.obs_dim
    for _ in range(env.episode_length):
        assert wrap.u == 4
        indicator = obs[d:]
        assert indicator.shape == (d,)
        assert int(np.sum(indicator)) == 1
        assert indicator[4] == 1.0
        assert obs[4] == 0.0
        obs, _, done, _ = wrap.step(np.zeros(env.action_dim, dtype=np.float32))
        if done:
            break
    assert done


def test_eq8_inverse_scores():
    returns = [10.0, 1.0, 10.0]
    p = update_pmiss(returns, l=0.0, eps=1e-8)
    assert p.shape == (3,)
    assert np.all(p > 0.0)
    np.testing.assert_allclose(p.sum(), 1.0, rtol=0, atol=1e-12)
    assert p[1] > p[0]
    assert p[1] > p[2]
    assert np.argmax(p) == 1


def test_uniform_baseline_does_not_change_pmiss():
    p0 = init_pmiss(5)
    p1 = update_pmiss_baseline(p0, returns=[1.0, 2.0, 3.0, 4.0, 0.0])
    np.testing.assert_array_equal(p0, p1)
    np.testing.assert_allclose(p1, np.full(5, 0.2))


def test_clipping_avoids_div_by_zero_and_inf():
    returns = np.array([-8.0, -0.5, 3.0])
    p = update_pmiss(returns, l=L_BOUND, eps=EPS)
    assert np.all(np.isfinite(p))
    assert not np.any(np.isnan(p))
    assert np.all(p > 0.0)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)
    # clipped negatives share the same inverse score 1/eps, so they dominate
    assert p[0] > p[2]
    assert p[1] > p[2]

    p_zero = update_pmiss(np.array([0.0, 0.0, 0.0]), l=0.0, eps=EPS)
    assert np.all(np.isfinite(p_zero))
    np.testing.assert_allclose(p_zero, np.full(3, 1.0 / 3.0), atol=1e-12)


def _tree_close(a, b):
    return jax.tree_util.tree_all(jax.tree.map(lambda x, y: jnp.allclose(x, y), a, b))


def test_sac_update_changes_params_and_actions_finite():
    obs_dim, action_dim = 12, 1
    agent = SAC(obs_dim=obs_dim, action_dim=action_dim, key=jax.random.PRNGKey(0), batch_size=32)
    actor_before = jax.tree.map(lambda x: np.array(x), agent.actor)
    q1_before = jax.tree.map(lambda x: np.array(x), agent.q1)

    rng = np.random.default_rng(1)
    batch = {
        "obs": rng.normal(size=(32, obs_dim)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(32, action_dim)).astype(np.float32),
        "reward": rng.normal(size=(32,)).astype(np.float32),
        "next_obs": rng.normal(size=(32, obs_dim)).astype(np.float32),
        "done": rng.integers(0, 2, size=(32,)).astype(np.float32),
    }
    ran = agent.update(batch)
    assert ran is True

    actor_same = _tree_close(actor_before, agent.actor)
    q1_same = _tree_close(q1_before, agent.q1)
    assert (not actor_same) or (not q1_same)

    obs = rng.normal(size=(obs_dim,)).astype(np.float32)
    action = agent.act(obs, deterministic=True)
    assert np.all(np.isfinite(action))
    action_stoch = agent.act(obs, deterministic=False)
    assert np.all(np.isfinite(action_stoch))


def test_toy_env_missing_critical_is_worse():
    env = ToyPointEnv(obs_dim=6, episode_length=32, seed=0)
    policy = make_readable_fn(env.action_dim)
    n_ep = 8
    ret_critical = []
    ret_dummy = []
    for i in range(n_ep):
        obs = env.reset(seed=100 + i)
        g = 0.0
        done = False
        while not done:
            a = policy(mask_observation(obs, 0))
            obs, r, done, _ = env.step(a)
            g += r
        ret_critical.append(g)
        obs = env.reset(seed=100 + i)
        g = 0.0
        done = False
        while not done:
            a = policy(mask_observation(obs, 5))
            obs, r, done, _ = env.step(a)
            g += r
        ret_dummy.append(g)
    assert float(np.mean(ret_critical)) < float(np.mean(ret_dummy))


def test_short_adversarial_training_concentrates_on_critical():
    result = train(
        total_steps=96,
        zeta=32,
        n_eval=3,
        adversarial=True,
        use_readable_policy=True,
        seed=2,
        obs_dim=6,
        episode_length=16,
    )
    p = result["p_miss"]
    d = 6
    uniform = 1.0 / d
    assert p[0] >= uniform
    assert p[0] > uniform
    assert p[0] == np.max(p)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-10)
