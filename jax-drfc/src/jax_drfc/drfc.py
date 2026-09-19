"""Algorithm 1 DRFC, complete-graph kernel (DG = 0).

Candidate hat a, epoch e. Between scans every agent pulls hat a. Every M rounds,
if idle: A = [K]; while |A| > 1 and budget Hscan is not exhausted, run one
synchronized balanced active-set block (h slots per active arm, independent
random order per agent) and eliminate a when max_b (Ghat_b,a - beta) > 0.
If A = {w} and w != hat a, switch. No local-change alarm. No local reset.

Records are shared instantly, so the equal-agent gap uses the full fresh window
of the current scan only.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from jax_drfc.certificates import PAPER_C, beta_radius, eliminate_once, pairwise_ghat_and_B
from jax_drfc.env import HeterogeneousBandit, aggregate_block, balanced_block_schedule


@dataclass
class DRFCConfig:
    n_agents: int
    n_arms: int
    horizon: int
    h: int = 2
    M: int = 16
    Hscan: int = 64
    delta: float = 0.05
    C: float = PAPER_C
    init_candidate: int = 0


@dataclass
class DRFCResult:
    candidate: int
    n_switches: int
    epoch: int
    t: int
    n_scans: int
    n_certified: int


def _active_arms(mask: Array) -> Array:
    return jnp.asarray([i for i in range(int(mask.shape[0])) if bool(mask[i])], dtype=jnp.int32)


def run_one_scan(
    env: HeterogeneousBandit,
    cfg: DRFCConfig,
    t: int,
    key: Array,
) -> tuple[int | None, int, bool, Array]:
    """Fresh active-set scan starting at time t. Returns (winner, t, certified, key)."""
    n_arms = cfg.n_arms
    active = jnp.ones((n_arms,), dtype=bool)
    remaining = int(cfg.Hscan)
    block_means: list[Array] = []
    block_active: list[Array] = []

    while int(active.sum()) > 1:
        n_active = int(active.sum())
        block_len = n_active * int(cfg.h)
        if remaining < block_len or t + block_len > env.horizon:
            break
        arms = _active_arms(active)
        key, k_sched, k_pull = jax.random.split(key, 3)
        schedule = balanced_block_schedule(k_sched, cfg.n_agents, arms, cfg.h)
        rewards, t = env.step_block(t, schedule, k_pull)
        remaining -= block_len
        means, _counts = aggregate_block(schedule, rewards, n_arms)
        block_means.append(means)
        block_active.append(active)

        stacked_means = jnp.stack(block_means, axis=0)
        stacked_active = jnp.stack(block_active, axis=0)
        ghat, n_blocks = pairwise_ghat_and_B(stacked_means, stacked_active)
        beta = beta_radius(
            cfg.n_agents,
            cfg.h,
            n_blocks,
            n_arms,
            cfg.horizon,
            cfg.delta,
            C=cfg.C,
        )
        active = eliminate_once(active, ghat, beta)

    certified = int(active.sum()) == 1
    winner = int(jnp.argmax(active.astype(jnp.int32))) if certified else None
    return winner, t, certified, key


def run_drfc(env: HeterogeneousBandit, cfg: DRFCConfig, key: Array) -> DRFCResult:
    """Run Algorithm 1 until env.horizon (or cfg.horizon, whichever is smaller)."""
    t = 0
    candidate = int(cfg.init_candidate)
    n_switches = 0
    epoch = 0
    n_scans = 0
    n_certified = 0
    horizon = min(int(cfg.horizon), env.horizon)
    n_agents = cfg.n_agents
    min_block = cfg.n_arms * cfg.h

    while t < horizon:
        can_scan = (
            t % cfg.M == 0
            and cfg.Hscan >= min_block
            and (horizon - t) >= min_block
        )
        if can_scan:
            n_scans += 1
            t_before = t
            winner, t, certified, key = run_one_scan(env, cfg, t, key)
            if t == t_before:
                key, k_play = jax.random.split(key)
                schedule = jnp.full((n_agents, 1), candidate, dtype=jnp.int32)
                _, t = env.step_block(t, schedule, k_play)
                continue
            if certified:
                n_certified += 1
                if winner is not None and winner != candidate:
                    candidate = winner
                    epoch += 1
                    n_switches += 1
            continue
        key, k_play = jax.random.split(key)
        schedule = jnp.full((n_agents, 1), candidate, dtype=jnp.int32)
        _, t = env.step_block(t, schedule, k_play)

    return DRFCResult(
        candidate=candidate,
        n_switches=n_switches,
        epoch=epoch,
        t=t,
        n_scans=n_scans,
        n_certified=n_certified,
    )
