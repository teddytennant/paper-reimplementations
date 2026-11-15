"""Training loop: SAC on masked observations, p_miss updated from inverse scores."""

from __future__ import annotations

import numpy as np

from jax_admissable.env import ToyPointEnv, readable_policy
from jax_admissable.missingness import mask_observation, sample_missing_index
from jax_admissable.pmiss import EPS, L_BOUND, ZETA, init_pmiss, update_pmiss, update_pmiss_baseline
from jax_admissable.sac import SAC


def episode_return(env, act_fn, u, seed=None):
    """Undiscounted sum of rewards with feature ``u`` missing for the episode."""
    obs = env.reset(seed=seed)
    done = False
    total = 0.0
    while not done:
        masked = mask_observation(obs, u)
        action = act_fn(masked)
        obs, reward, done, _ = env.step(action)
        total += float(reward)
    return total


def estimate_adversarial_scores(env, act_fn, n_episodes=4, seed=0):
    """a^pi_u: mean undiscounted return for every missing index u (paper Sec. 3)."""
    scores = np.zeros(env.obs_dim, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for u in range(env.obs_dim):
        rets = []
        for _ in range(int(n_episodes)):
            ep_seed = int(rng.integers(0, 2**31 - 1))
            rets.append(episode_return(env, act_fn, u, seed=ep_seed))
        scores[u] = float(np.mean(rets))
    return scores


def make_actor_fn(sac):
    def act_fn(masked_obs):
        return sac.act(masked_obs, deterministic=True)

    return act_fn


def make_readable_fn(action_dim):
    def act_fn(masked_obs):
        return readable_policy(masked_obs, action_dim=action_dim)

    return act_fn


def train(
    total_steps=400,
    zeta=200,
    n_eval=3,
    adversarial=True,
    use_readable_policy=False,
    seed=0,
    obs_dim=6,
    episode_length=32,
    sac=None,
    env=None,
):
    """Run a short adversarial-missingness training loop.

    Every ``zeta`` environment steps, adversarial scores are estimated
    for each feature and ``p_miss`` is updated with Eq. 8 (unless
    ``adversarial`` is False, in which case it stays uniform).
    """
    rng = np.random.default_rng(seed)
    if env is None:
        env = ToyPointEnv(obs_dim=obs_dim, episode_length=episode_length, seed=seed)
    d = env.obs_dim
    if sac is None:
        sac = SAC(obs_dim=2 * d, action_dim=env.action_dim, key=None)
    p_miss = init_pmiss(d)
    act_fn = make_readable_fn(env.action_dim) if use_readable_policy else make_actor_fn(sac)

    obs = env.reset(seed=seed)
    u = sample_missing_index(p_miss, rng)
    scores = None
    for step in range(int(total_steps)):
        masked = mask_observation(obs, u)
        if use_readable_policy:
            action = act_fn(masked)
        else:
            action = sac.act(masked, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
        next_masked = mask_observation(next_obs, u)
        sac.store(masked, action, reward, next_masked, done)
        if not use_readable_policy:
            sac.update()
        obs = next_obs
        if done:
            obs = env.reset()
            u = sample_missing_index(p_miss, rng)
        if zeta > 0 and step > 0 and (step + 1) % int(zeta) == 0:
            scores = estimate_adversarial_scores(env, act_fn, n_episodes=n_eval, seed=seed + step)
            if adversarial:
                p_miss = update_pmiss(scores, l=L_BOUND, eps=EPS)
            else:
                p_miss = update_pmiss_baseline(p_miss, scores)
    return {
        "p_miss": p_miss,
        "scores": scores,
        "sac": sac,
        "env": env,
        "zeta": zeta if zeta else ZETA,
    }


def smoke_train():
    """Hundreds of env steps; print-ready p_miss."""
    result = train(
        total_steps=400,
        zeta=100,
        n_eval=2,
        adversarial=True,
        use_readable_policy=True,
        seed=1,
    )
    return result["p_miss"]
