"""Heterogeneous multi-agent bandit toys, including the paper's cancellation story.

N agents, K arms. At global time t, agent i pulling arm a gets a reward with mean
mu[t, i, a]. The network (equal-agent) mean of arm a is (1/N) sum_i mu[t, i, a].

Cancellation construction (paper, two-arm case):
    mu[i, 0] = 1/2 + Delta + sigma[i] * eps
    mu[i, 1] = 1/2 - sigma[i] * eps
with sigma in {-1, +1} and sum_i sigma[i] = 0. Flipping all signs leaves the
equal-agent means unchanged (S_dec = 0) while every local mean jumps (S_loc large).

Complete-graph kernel: agents share records instantly (DG = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


def cancellation_mean_path(
    horizon: int,
    n_agents: int = 2,
    n_arms: int = 2,
    delta: float = 0.25,
    eps: float = 0.20,
    period: int = 16,
    switch_time: int | None = None,
) -> Array:
    """Build mu[t, i, a] for the cancellation toy.

    First n_agents//2 agents get sigma = +1, the rest sigma = -1, so the signs
    cancel when n_agents is even. Signs flip every `period` rounds.

    If `switch_time` is set, arms 0 and 1 swap from that time onward, so the
    common best arm changes (S_dec = 1).
    """
    t = jnp.arange(horizon)
    flip = jnp.where((t // period) % 2 == 0, 1.0, -1.0)
    sigma0 = jnp.ones((n_agents,), dtype=jnp.float32)
    sigma0 = sigma0.at[n_agents // 2 :].set(-1.0)
    sigma = flip[:, None] * sigma0[None, :]
    mu = jnp.zeros((horizon, n_agents, n_arms), dtype=jnp.float32)
    mu = mu.at[:, :, 0].set(0.5 + delta + sigma * eps)
    if n_arms >= 2:
        mu = mu.at[:, :, 1].set(0.5 - sigma * eps)
    if n_arms >= 3:
        mu = mu.at[:, :, 2:].set(0.25)
    if switch_time is not None and 0 <= switch_time < horizon:
        tail = mu[switch_time:]
        swapped = tail.at[:, :, [0, 1]].set(tail[:, :, [1, 0]])
        mu = jnp.concatenate([mu[:switch_time], swapped], axis=0)
    return mu


def count_s_loc_s_dec(mean_path: Array, atol: float = 1e-6) -> tuple[int, int]:
    """S_loc: times any local mean jumps. S_dec: times the equal-agent best arm jumps."""
    d = jnp.abs(mean_path[1:] - mean_path[:-1])
    s_loc = int(jnp.any(d > atol, axis=(1, 2)).sum())
    eq = mean_path.mean(axis=1)
    best = jnp.argmax(eq, axis=1)
    s_dec = int((best[1:] != best[:-1]).sum())
    return s_loc, s_dec


def balanced_block_schedule(
    key: Array, n_agents: int, active_arms: Array, h: int
) -> Array:
    """Independent random order per agent of each active arm repeated h times.

    Returns int32 schedule of shape (n_agents, n_active * h).
    """
    active_arms = jnp.asarray(active_arms, dtype=jnp.int32)
    base = jnp.repeat(active_arms, int(h))
    keys = jax.random.split(key, int(n_agents))
    return jax.vmap(lambda k: jax.random.permutation(k, base))(keys)


def aggregate_block(schedule: Array, rewards: Array, n_arms: int) -> tuple[Array, Array]:
    """Per-agent empirical means and counts from a balanced block.

    schedule, rewards: (N, L). Returns means, counts each (N, K).
    Inactive arms have mean 0 and count 0.
    """
    n_agents, length = schedule.shape
    agent_ix = jnp.arange(n_agents)[:, None]
    sums = jnp.zeros((n_agents, n_arms), dtype=rewards.dtype)
    counts = jnp.zeros((n_agents, n_arms), dtype=rewards.dtype)
    sums = sums.at[agent_ix, schedule].add(rewards)
    counts = counts.at[agent_ix, schedule].add(jnp.ones_like(rewards))
    means = jnp.where(counts > 0, sums / counts, 0.0)
    return means, counts


@dataclass
class HeterogeneousBandit:
    """Time-varying heterogeneous Bernoulli (or deterministic) bandit."""

    mean_path: Array
    noise: str = "none"

    def __post_init__(self) -> None:
        self.mean_path = jnp.asarray(self.mean_path, dtype=jnp.float32)
        if self.noise not in ("none", "bernoulli"):
            raise ValueError(f"unknown noise {self.noise}")

    @property
    def horizon(self) -> int:
        return int(self.mean_path.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.mean_path.shape[1])

    @property
    def n_arms(self) -> int:
        return int(self.mean_path.shape[2])

    def means_at(self, t: int) -> Array:
        t = min(max(t, 0), self.horizon - 1)
        return self.mean_path[t]

    def step_block(self, t: int, schedule: Array, key: Array) -> tuple[Array, int]:
        """Synchronized pulls: agent i in slot s plays schedule[i, s].

        Wall-clock length is L = schedule.shape[1] (DG = 0, complete graph).
        """
        schedule = jnp.asarray(schedule, dtype=jnp.int32)
        n_agents, length = schedule.shape
        agent_ix = jnp.arange(n_agents)
        keys = jax.random.split(key, length)
        cols = []
        for s in range(length):
            tt = t + s
            if tt >= self.horizon:
                cols.append(jnp.zeros((n_agents,), dtype=jnp.float32))
                continue
            p = self.mean_path[tt][agent_ix, schedule[:, s]]
            if self.noise == "none":
                cols.append(p)
            else:
                draw = jax.random.bernoulli(keys[s], jnp.clip(p, 0.0, 1.0))
                cols.append(draw.astype(jnp.float32))
        rewards = jnp.stack(cols, axis=1)
        return rewards, t + length


def make_cancellation_env(
    horizon: int = 80,
    n_agents: int = 2,
    n_arms: int = 2,
    delta: float = 0.25,
    eps: float = 0.20,
    period: int = 16,
    switch_time: int | None = None,
    noise: str = "none",
) -> HeterogeneousBandit:
    path = cancellation_mean_path(
        horizon=horizon,
        n_agents=n_agents,
        n_arms=n_arms,
        delta=delta,
        eps=eps,
        period=period,
        switch_time=switch_time,
    )
    return HeterogeneousBandit(mean_path=path, noise=noise)
