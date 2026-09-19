"""Algorithm 2 DRFC-Seq kernel (complete graph, DG = 0).

Every delta_p rounds, one balanced all-arm probe block. Sliding window Q of the
last W fully-flooded blocks (every probe is fully flooded immediately). Act only
when |Q| = W. Switch if there exists b != hat a with

    (muhat_b - muhat_hat_a) - (r_{n_b} + r_{n_hat_a}) > 0

then hat a <- argmax_a muhat_a. Anytime-valid radius as in certificates.py
(a0 = 1, C0 = 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from jax_drfc.certificates import SEQ_A0, SEQ_C0, anytime_radius, equal_agent_means
from jax_drfc.env import HeterogeneousBandit, aggregate_block, balanced_block_schedule


@dataclass
class DRFCSeqConfig:
    n_agents: int
    n_arms: int
    horizon: int
    W: int = 4
    delta_p: int = 8
    h: int = 2
    alpha: float = 0.05
    a0: float = SEQ_A0
    C0: float = SEQ_C0
    init_candidate: int = 0


@dataclass
class DRFCSeqResult:
    candidate: int
    n_switches: int
    t: int
    n_probes: int
    window_len: int


def seq_switch_decision(
    muhat: Array,
    candidate: int,
    n: Array | int | float,
    s: Array | int | float,
    n_agents: int,
    n_arms: int,
    alpha: float,
    a0: float = SEQ_A0,
    C0: float = SEQ_C0,
) -> tuple[int, bool]:
    """Switch if some b has a positive lower-confidence gap vs the candidate.

    n is the per-agent sample count (W h on a full balanced window).
    s is the 1-based window-start block index.
    Returns (new_candidate, did_switch).
    """
    r = anytime_radius(n, s, n_agents, n_arms, alpha, a0=a0, C0=C0)
    lc = (muhat - muhat[candidate]) - (r + r)
    lc = lc.at[candidate].set(-jnp.inf)
    if bool(jnp.any(lc > 0)):
        return int(jnp.argmax(muhat)), True
    return int(candidate), False


def run_drfc_seq(env: HeterogeneousBandit, cfg: DRFCSeqConfig, key: Array) -> DRFCSeqResult:
    t = 0
    candidate = int(cfg.init_candidate)
    n_switches = 0
    n_probes = 0
    horizon = min(int(cfg.horizon), env.horizon)
    n_agents = cfg.n_agents
    n_arms = cfg.n_arms
    probe_len = n_arms * cfg.h
    all_arms = jnp.arange(n_arms, dtype=jnp.int32)
    window_means: list[Array] = []
    block_index = 0

    while t < horizon:
        if t % cfg.delta_p == 0 and t + probe_len <= horizon:
            key, k_sched, k_pull = jax.random.split(key, 3)
            schedule = balanced_block_schedule(k_sched, n_agents, all_arms, cfg.h)
            rewards, t = env.step_block(t, schedule, k_pull)
            means, _counts = aggregate_block(schedule, rewards, n_arms)
            window_means.append(means)
            block_index += 1
            n_probes += 1
            if len(window_means) > cfg.W:
                window_means.pop(0)
            if len(window_means) == cfg.W:
                stacked = jnp.stack(window_means, axis=0)
                muhat = equal_agent_means(stacked.mean(axis=0))
                n = cfg.W * cfg.h
                s = block_index - cfg.W + 1
                new_c, switched = seq_switch_decision(
                    muhat,
                    candidate,
                    n,
                    s,
                    n_agents,
                    n_arms,
                    cfg.alpha,
                    a0=cfg.a0,
                    C0=cfg.C0,
                )
                if switched:
                    candidate = new_c
                    n_switches += 1
            continue
        key, k_play = jax.random.split(key)
        schedule = jnp.full((n_agents, 1), candidate, dtype=jnp.int32)
        _, t = env.step_block(t, schedule, k_play)

    return DRFCSeqResult(
        candidate=candidate,
        n_switches=n_switches,
        t=t,
        n_probes=n_probes,
        window_len=len(window_means),
    )
