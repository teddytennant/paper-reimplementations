"""Observation masking for adversarial missingness (paper Section 3).

Missing coordinates are replaced by zeros. A binary indicator vector
v with v_i = I(i in u) is concatenated, so the agent sees a vector of
length 2d. For k=1, u is a single feature index, held fixed for an
entire episode.
"""

from __future__ import annotations

import numpy as np

from jax_admissable.pmiss import init_pmiss, sample_missing_index


def mask_observation(s, u):
    """Replace feature ``u`` with 0 and concatenate a one-hot indicator.

    Parameters
    ----------
    s : array-like, shape (d,) or (..., d)
        Full observation.
    u : int
        Index of the missing feature (k=1).

    Returns
    -------
    ndarray
        ``concat(masked_s, v)`` with length 2d along the last axis.
        ``masked_s[u] == 0`` and ``v`` is 1 at ``u`` and 0 elsewhere.
    """
    s = np.asarray(s, dtype=np.float32)
    if s.ndim == 0:
        raise ValueError("observation must have at least one dimension")
    d = s.shape[-1]
    u = int(u)
    if u < 0 or u >= d:
        raise ValueError(f"missing index u={u} is outside [0, {d})")
    masked = np.array(s, dtype=np.float32, copy=True)
    masked[..., u] = 0.0
    v = np.zeros(s.shape, dtype=np.float32)
    v[..., u] = 1.0
    return np.concatenate([masked, v], axis=-1)


class MissingnessWrapper:
    """Wraps an env so one sampled feature is missing for a whole episode.

    ``u`` is drawn from ``p_miss`` at ``reset`` and then kept constant
    until the next reset, matching the paper's globally-fixed mask.
    """

    def __init__(self, env, p_miss=None, rng=None):
        self.env = env
        self.d = int(env.obs_dim)
        self.obs_dim = 2 * self.d
        self.action_dim = int(env.action_dim)
        if p_miss is None:
            p_miss = init_pmiss(self.d)
        self.p_miss = np.asarray(p_miss, dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.u = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.u = sample_missing_index(self.p_miss, self.rng)
        obs = self.env.reset(seed=seed)
        return mask_observation(obs, self.u), int(self.u)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return mask_observation(obs, self.u), reward, done, info
