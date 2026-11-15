"""Toy continuous environment with a planted critical observation coordinate.

This environment is **not** in the paper. The paper uses MuJoCo Hopper,
Walker, and Ant. Those simulators are intentionally not depended on here.

Observation is in R^d (default d=6):
  obs[0] — target action / target velocity (the critical feature)
  obs[1] — integrated position (unused by the readable policy)
  obs[2:] — i.i.d. Gaussian noise

Action is in [-1, 1]^{action_dim}. Reward is large only if the agent
uses obs[0]: ``r = action_dim - ||action - target * ones||^2``. Missing
obs[0] therefore tanks return relative to missing a noise coordinate,
under a policy that copies obs[0] into the action. The shift keeps
typical returns above the paper's clip ``l=0``.
"""

from __future__ import annotations

import numpy as np


class ToyPointEnv:
    """1D point with a planted critical observation coordinate."""

    def __init__(
        self,
        obs_dim=6,
        action_dim=1,
        episode_length=32,
        dt=0.1,
        seed=0,
    ):
        if obs_dim < 1:
            raise ValueError("obs_dim must be at least 1")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.episode_length = int(episode_length)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.x = 0.0
        self.target = 0.0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.x = 0.0
        sign = float(self.rng.choice([-1.0, 1.0]))
        self.target = sign * float(self.rng.uniform(0.5, 1.0))
        return self._obs()

    def _obs(self):
        obs = self.rng.normal(0.0, 0.05, size=self.obs_dim).astype(np.float32)
        obs[0] = np.float32(self.target)
        if self.obs_dim > 1:
            obs[1] = np.float32(self.x)
        return obs

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.action_dim:
            a = np.resize(a, self.action_dim)
        a = np.clip(a, -1.0, 1.0)
        self.x = self.x + float(a[0]) * self.dt
        self.t += 1
        target_vec = np.full(self.action_dim, self.target, dtype=np.float32)
        # Shifted so typical returns are positive (paper clips a_u at l=0).
        # Perfect tracking yields +action_dim per step; missing obs[0] is worse.
        reward = float(self.action_dim) - float(np.sum((a - target_vec) ** 2))
        done = self.t >= self.episode_length
        return self._obs(), reward, done, {"target": self.target}

    def sample_action(self):
        return self.rng.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)


def readable_policy(masked_obs, action_dim=1):
    """Hardcoded policy that copies the (possibly masked) obs[0] into the action.

    Not a trained SAC actor. Used to test that missing the critical
    feature is uniquely harmful.
    """
    target = float(np.asarray(masked_obs).reshape(-1)[0])
    a = np.full(action_dim, np.clip(target, -1.0, 1.0), dtype=np.float32)
    return a
