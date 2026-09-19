"""Equal-agent gaps and radii for DRFC / DRFC-Seq.

Fresh window W of B synchronized balanced blocks, h pulls per active arm per
agent. Equal-agent gap (paper):

    ghat_i,a,b(W) = (1/B) sum_ell (mean of a minus mean of b on block ell, agent i)
    Ghat_a,b(W)   = (1/N) sum_i ghat_i,a,b(W)

using only blocks in which both a and b are active.

Radius (paper: O-tilde of this; we use a concrete C):

    beta(W) = C * sqrt( log(K^2 T^3 / delta) / (N h B) )

with natural log and C = 8 (paper states C <= 8). Lower-confidence gap is
Ghat_b,a - beta. Eliminate a when max_{b in A\\{a}} (Ghat_b,a - beta) > 0.

DRFC-Seq anytime-valid radius (documented simplification a0 = 1, C0 = 1):

    r_n = a0 * sqrt( u_{n,s} / (N n) )
    u_{n,s} = log log(e n) + log(C0 K (K-1) s^2 / alpha)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

PAPER_C = 8.0
SEQ_A0 = 1.0
SEQ_C0 = 1.0


def beta_radius(
    n_agents: Array | int | float,
    h: Array | int | float,
    n_blocks: Array | int | float,
    n_arms: Array | int | float,
    horizon: Array | int | float,
    delta: Array | float,
    C: float = PAPER_C,
) -> Array:
    """beta(W) = C * sqrt( log(K^2 T^3 / delta) / (N h B) )."""
    n_agents = jnp.asarray(n_agents, dtype=jnp.float32)
    h = jnp.asarray(h, dtype=jnp.float32)
    n_blocks = jnp.asarray(n_blocks, dtype=jnp.float32)
    n_arms = jnp.asarray(n_arms, dtype=jnp.float32)
    horizon = jnp.asarray(horizon, dtype=jnp.float32)
    delta = jnp.asarray(delta, dtype=jnp.float32)
    num = jnp.log((n_arms ** 2) * (horizon ** 3) / delta)
    den = n_agents * h * jnp.maximum(n_blocks, 1.0)
    return jnp.asarray(C, dtype=jnp.float32) * jnp.sqrt(num / den)


def anytime_radius(
    n: Array | int | float,
    s: Array | int | float,
    n_agents: Array | int | float,
    n_arms: Array | int | float,
    alpha: Array | float,
    a0: float = SEQ_A0,
    C0: float = SEQ_C0,
) -> Array:
    """r_n = a0 * sqrt( u_{n,s} / (N n) ) with the paper's u_{n,s}."""
    n = jnp.maximum(jnp.asarray(n, dtype=jnp.float32), 1.0)
    s = jnp.maximum(jnp.asarray(s, dtype=jnp.float32), 1.0)
    n_agents = jnp.asarray(n_agents, dtype=jnp.float32)
    n_arms = jnp.asarray(n_arms, dtype=jnp.float32)
    alpha = jnp.asarray(alpha, dtype=jnp.float32)
    u = jnp.log(jnp.log(jnp.e * n)) + jnp.log(
        C0 * n_arms * (n_arms - 1.0) * (s ** 2) / alpha
    )
    u = jnp.maximum(u, 0.0)
    return jnp.asarray(a0, dtype=jnp.float32) * jnp.sqrt(u / (n_agents * n))


def equal_agent_means(per_agent_means: Array) -> Array:
    """Average empirical means across agents. per_agent_means: (N, K) -> (K,)."""
    return per_agent_means.mean(axis=0)


def pairwise_ghat_and_B(
    block_means: Array, block_active: Array
) -> tuple[Array, Array]:
    """Equal-agent Ghat[a, b] and co-active block counts B[a, b].

    block_means: (B, N, K) per-agent empirical means on each fresh block.
    block_active: (B, K) bool, whether the arm was in that block's active set.
    Ghat[a, b] = mean over co-active blocks of (equal-agent mean_a - mean_b).
    """
    eq = block_means.mean(axis=1)
    ghat_block = eq[:, :, None] - eq[:, None, :]
    both = block_active[:, :, None] & block_active[:, None, :]
    w = both.astype(ghat_block.dtype)
    n_blocks = w.sum(axis=0)
    ghat = jnp.sum(ghat_block * w, axis=0) / jnp.maximum(n_blocks, 1.0)
    return ghat, n_blocks


def lower_confidence_gap(ghat: Array, beta: Array) -> Array:
    """Ghat - beta, the lower-confidence gap used for elimination / switching."""
    return ghat - beta


def eliminate_once(active: Array, ghat: Array, beta: Array) -> Array:
    """Drop every active a with max_{b in A\\{a}} (Ghat[b, a] - beta) > 0.

    The empirical equal-agent best cannot be dropped (its LC gaps are negative
    whenever beta > 0), so the active set cannot empty from a pair comparison.
    """
    n_arms = active.shape[0]
    lc = lower_confidence_gap(ghat, beta)
    b_ok = active[:, None]
    a_ok = active[None, :]
    not_diag = ~jnp.eye(n_arms, dtype=bool)
    valid = b_ok & a_ok & not_diag
    lc_masked = jnp.where(valid, lc, jnp.asarray(-jnp.inf, dtype=lc.dtype))
    max_lc = lc_masked.max(axis=0)
    drop = active & (max_lc > 0)
    return active & ~drop
