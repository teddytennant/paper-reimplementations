"""Hierarchical delayed match-to-sample environment.

Latent context c in {0, ..., M-1} maps a circle angle θ to a 2D target by
rotation 2π c / M. Context is held for a block of L ~ Uniform(3, 25) trials.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Config:
    Nx: int = 16
    Nh: int = 8
    M: int = 2
    Nu: int = 3
    Ny: int = 2
    hidden: int = 8
    tau_x: float = 0.25
    tau_h: float = 0.8
    n_fix: int = 2
    n_stim: int = 3
    n_delay: int = 2
    n_resp: int = 3
    block_len_min: int = 3
    block_len_max: int = 25
    reward_threshold: float = 0.5
    p_star: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    h_std: float = 0.01

    @property
    def T(self) -> int:
        return self.n_fix + self.n_stim + self.n_delay + self.n_resp

    @property
    def No(self) -> int:
        return self.M + 1

    @property
    def resp_start(self) -> int:
        return self.n_fix + self.n_stim + self.n_delay


def target_xy(theta, context, M):
    """Map stimulus angle through latent context: θ ↦ θ + 2π c / M."""
    theta = jnp.asarray(theta, dtype=jnp.float32)
    angle = theta + (2.0 * jnp.pi * jnp.asarray(context, dtype=jnp.float32) / M)
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)])


def stimulus_xy(theta):
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)])


def build_trial(theta, context, cfg: Config):
    """One trial of fixation / stimulus / delay / go.

    Returns
    -------
    u : (T, 3)  [x, y, trigger]
    y_tgt : (T, 2)  origin except the response window
    mask : (T,)  1 on the response window
    target : (2,)  context-mapped reach location
    """
    T = cfg.T
    stim = stimulus_xy(theta)
    target = target_xy(theta, context, cfg.M)
    t = jnp.arange(T)
    in_stim = (t >= cfg.n_fix) & (t < cfg.n_fix + cfg.n_stim)
    in_resp = t >= cfg.resp_start
    xy = jnp.where(in_stim[:, None], stim[None, :], jnp.zeros((T, 2)))
    trig = in_resp.astype(xy.dtype)
    u = jnp.concatenate([xy, trig[:, None]], axis=-1)
    y_tgt = jnp.where(in_resp[:, None], target[None, :], jnp.zeros((T, 2)))
    mask = in_resp.astype(xy.dtype)
    return u, y_tgt, mask, target


def reach_reward(y, target, threshold=0.5):
    """1 if ||y - target|| is small, else 0."""
    dist = jnp.linalg.norm(jnp.asarray(y) - jnp.asarray(target))
    return jnp.where(dist < threshold, 1.0, 0.0)


def reach_mse(y, target):
    err = jnp.asarray(y) - jnp.asarray(target)
    return jnp.mean(err * err)


def stochastic_reward(reach_ok, pi_index, context, M, p_star, key):
    """Bandit reward from Appendix A / Sec. II-B.

    Correct context and successful reach: Bern(p*).
    Incorrect context: Bern((1-p*)/(M-1)).
    Correct context but failed reach: 0.
    """
    correct_ctx = jnp.asarray(pi_index) == jnp.asarray(context)
    p_wrong = (1.0 - p_star) / jnp.maximum(M - 1, 1)
    p = jnp.where(correct_ctx, jnp.where(reach_ok, p_star, 0.0), p_wrong)
    return jax.random.bernoulli(key, p).astype(jnp.float32)


def tile_contexts(context_ids, lengths):
    """Concatenate blocks of given contexts and lengths."""
    parts = [np.full(int(L), int(c), dtype=np.int32) for c, L in zip(context_ids, lengths)]
    return np.concatenate(parts)


def sample_context_sequence(key, n_trials, M, lo=3, hi=25):
    """Blockwise latent contexts, L ~ Uniform(lo, hi) inclusive.

    Returns contexts (n_trials,), block_ids (n_trials,).
    """
    contexts = np.zeros((n_trials,), dtype=np.int32)
    block_ids = np.zeros((n_trials,), dtype=np.int32)
    t = 0
    b = 0
    while t < n_trials:
        key, kL, kC = jax.random.split(key, 3)
        L = int(jax.random.randint(kL, (), lo, hi + 1))
        c = int(jax.random.randint(kC, (), 0, M))
        L = min(L, n_trials - t)
        contexts[t : t + L] = c
        block_ids[t : t + L] = b
        t += L
        b += 1
    return contexts, block_ids


def make_nrnn_batch(key, cfg: Config, batch_size, contexts=None):
    """Batch of trials with teacher π = one-hot(true context)."""
    if contexts is None:
        key, kc = jax.random.split(key)
        contexts = jax.random.randint(kc, (batch_size,), 0, cfg.M)
    else:
        contexts = jnp.asarray(contexts, dtype=jnp.int32)
    key, kt = jax.random.split(key)
    thetas = jax.random.uniform(kt, (batch_size,), minval=0.0, maxval=2.0 * jnp.pi)

    def one(theta, context):
        u, y_tgt, mask, target = build_trial(theta, context, cfg)
        pi = jax.nn.one_hot(context, cfg.M)
        return u, y_tgt, mask, pi, target

    u, y_tgt, mask, pi, target = jax.vmap(one)(thetas, contexts)
    return {
        "u": u,
        "y_tgt": y_tgt,
        "mask": mask,
        "pi": pi,
        "target": target,
        "context": contexts,
        "theta": thetas,
    }
