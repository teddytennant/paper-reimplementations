"""RouteRelay (arXiv:2609.07306), following the paper's numbered equations.

Mandatory global candidates (sink / recent chunks) are omitted from the core
ledger. Forced-anchor strides are also omitted; hysteresis still holds a
just-refreshed row for one layer.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Ledger(NamedTuple):
    """Per-row route ledger at the last anchor, eq 4: (R_i^a, C_i^a, a).

    routes: [n_q, k] selected key-chunk ids.
    sentinels: [n_q, m + r] near-miss followed by random sentinels.
    ages: [n_q] layers since the last full refresh (0 on a just-written row).
    """

    routes: jax.Array
    sentinels: jax.Array
    ages: jax.Array


class RelayInfo(NamedTuple):
    reuse: jax.Array
    alpha: jax.Array
    beta: jax.Array
    n_refreshed: jax.Array
    n_score_pairs: jax.Array


def chunk_scores(q_c: jax.Array, k_c: jax.Array) -> jax.Array:
    """Eq 1: S^ell = Q_c^ell (K_c^ell)^T."""
    return q_c @ k_c.T


def topk_routes(scores: jax.Array, k: int) -> jax.Array:
    """Eq 2: R_i = TopK_j(S_ij, k). Indices, highest score first."""
    return jax.lax.top_k(scores, k)[1].astype(jnp.int32)


def _omit_mask(n_cols: int, ids: jax.Array) -> jax.Array:
    """False at listed column ids, True elsewhere. ids is [n_q, w]."""
    n_q = ids.shape[0]
    mask = jnp.ones((n_q, n_cols), dtype=bool)
    return mask.at[jnp.arange(n_q)[:, None], ids].set(False)


def near_miss_sentinels(scores: jax.Array, routes: jax.Array, m: int) -> jax.Array:
    """The m highest-scoring chunks immediately below the top-k cutoff."""
    n_q = scores.shape[0]
    if m == 0:
        return jnp.zeros((n_q, 0), dtype=jnp.int32)
    omitted = _omit_mask(scores.shape[-1], routes)
    packed = jnp.where(omitted, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype))
    return jax.lax.top_k(packed, m)[1].astype(jnp.int32)


def random_sentinels(
    key: jax.Array,
    n_k: int,
    routes: jax.Array,
    near_miss: jax.Array,
    r: int,
) -> jax.Array:
    """r uniform draws without replacement from the residual omitted pool."""
    n_q = routes.shape[0]
    if r == 0:
        return jnp.zeros((n_q, 0), dtype=jnp.int32)
    taken = jnp.concatenate([routes, near_miss], axis=1)
    residual = _omit_mask(n_k, taken)
    gumbel = jax.random.gumbel(key, (n_q, n_k), dtype=jnp.float32)
    packed = jnp.where(residual, gumbel, jnp.asarray(-jnp.inf, dtype=gumbel.dtype))
    return jax.lax.top_k(packed, r)[1].astype(jnp.int32)


def build_ledger(
    scores: jax.Array,
    k: int,
    m: int,
    r: int,
    rng_key: jax.Array,
) -> Ledger:
    """Full top-k plus near-miss and random sentinels at an anchor layer."""
    n_q, n_k = scores.shape
    routes = topk_routes(scores, k)
    near = near_miss_sentinels(scores, routes, m)
    rand = random_sentinels(rng_key, n_k, routes, near, r)
    if m == 0 and r == 0:
        sentinels = jnp.zeros((n_q, 0), dtype=jnp.int32)
    else:
        sentinels = jnp.concatenate([near, rand], axis=1)
    ages = jnp.zeros((n_q,), dtype=jnp.int32)
    return Ledger(routes=routes, sentinels=sentinels, ages=ages)


def candidate_set(routes: jax.Array, sentinels: jax.Array) -> jax.Array:
    """Eq 5: A_i = R_i^a union C_i^a, stored as a concatenated index tensor."""
    return jnp.concatenate([routes, sentinels], axis=1)


def _gather_scores(q_c: jax.Array, k_c: jax.Array, idx: jax.Array) -> jax.Array:
    """Score listed key chunks per query row. idx is [n_q, c]."""
    if idx.shape[1] == 0:
        return jnp.zeros((q_c.shape[0], 0), dtype=q_c.dtype)
    return jnp.einsum("nd,ncd->nc", q_c, k_c[idx])


def rescore_candidates(
    q_c: jax.Array,
    k_c: jax.Array,
    routes: jax.Array,
    sentinels: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Score A_i, then alpha = min_R S (eq 6) and beta = max_C S (eq 7)."""
    cand = candidate_set(routes, sentinels)
    scores = _gather_scores(q_c, k_c, cand)
    k = routes.shape[1]
    route_scores = scores[:, :k]
    sent_scores = scores[:, k:]
    alpha = route_scores.min(axis=1)
    if sent_scores.shape[1] == 0:
        beta = jnp.full((q_c.shape[0],), -jnp.inf, dtype=scores.dtype)
    else:
        beta = sent_scores.max(axis=1)
    return cand, scores, alpha, beta


def reuse_mask(
    alpha: jax.Array,
    beta: jax.Array,
    gamma: jax.Array | float,
    ages: jax.Array,
    hysteresis_horizon: int = 1,
) -> jax.Array:
    """Eq 8: reuse when alpha > beta + gamma, else refresh.

    A just-refreshed row (age < hysteresis_horizon) is held for one layer.
    """
    stable = alpha > (beta + gamma)
    held = ages < hysteresis_horizon
    return stable | held


def _rebuild_row(
    q_row: jax.Array,
    k_c: jax.Array,
    key: jax.Array,
    k: int,
    m: int,
    r: int,
) -> tuple[jax.Array, jax.Array]:
    """Full eq-1 row, then a new route and sentinel set."""
    scores = k_c @ q_row
    n_k = scores.shape[0]
    routes = jax.lax.top_k(scores, k)[1].astype(jnp.int32)
    omit = jnp.ones((n_k,), dtype=bool).at[routes].set(False)
    if m == 0:
        near = jnp.zeros((0,), dtype=jnp.int32)
    else:
        packed = jnp.where(omit, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype))
        near = jax.lax.top_k(packed, m)[1].astype(jnp.int32)
        omit = omit.at[near].set(False)
    if r == 0:
        rand = jnp.zeros((0,), dtype=jnp.int32)
    else:
        gumbel = jax.random.gumbel(key, (n_k,), dtype=jnp.float32)
        packed_g = jnp.where(omit, gumbel, jnp.asarray(-jnp.inf, dtype=gumbel.dtype))
        rand = jax.lax.top_k(packed_g, r)[1].astype(jnp.int32)
    sentinels = jnp.concatenate([near, rand], axis=0)
    return routes, sentinels


def relay_layer(
    q_c: jax.Array,
    k_c: jax.Array,
    ledger: Ledger,
    *,
    k: int,
    m: int,
    r: int,
    gamma: float = 0.0,
    rng_key: jax.Array,
    hysteresis_horizon: int = 1,
) -> tuple[Ledger, RelayInfo]:
    """Candidate rescoring plus row-selective refresh.

    Every row scores the fixed-width candidate set. Rows that fail eq 8 and
    are outside the hysteresis hold recompute a full score row via vmap.
    """
    n_q = q_c.shape[0]
    n_k = k_c.shape[0]
    _, _, alpha, beta = rescore_candidates(q_c, k_c, ledger.routes, ledger.sentinels)
    reuse = reuse_mask(alpha, beta, gamma, ledger.ages, hysteresis_horizon)
    need = jnp.logical_not(reuse)
    keys = jax.random.split(rng_key, n_q)

    def per_row(need_i, q_i, route_i, sent_i, age_i, key_i):
        def refresh():
            new_rt, new_st = _rebuild_row(q_i, k_c, key_i, k, m, r)
            return new_rt, new_st, jnp.int32(0)

        def keep():
            return route_i, sent_i, age_i + jnp.int32(1)

        return jax.lax.cond(need_i, refresh, keep)

    routes, sentinels, ages = jax.vmap(per_row)(
        need, q_c, ledger.routes, ledger.sentinels, ledger.ages, keys
    )
    n_refreshed = need.sum().astype(jnp.int32)
    cand_width = jnp.int32(ledger.routes.shape[1] + ledger.sentinels.shape[1])
    n_score_pairs = (jnp.int32(n_q) * cand_width + n_refreshed * jnp.int32(n_k)).astype(
        jnp.int32
    )
    info = RelayInfo(
        reuse=reuse,
        alpha=alpha,
        beta=beta,
        n_refreshed=n_refreshed,
        n_score_pairs=n_score_pairs,
    )
    return Ledger(routes=routes, sentinels=sentinels, ages=ages), info


def anchor_margin(scores: jax.Array, routes: jax.Array) -> jax.Array:
    """Eq 9: Delta_i^a = min_{j in R} S_ij^a - max_{j not in R} S_ij^a."""
    n_k = scores.shape[-1]
    selected = jnp.take_along_axis(scores, routes, axis=1)
    omitted = _omit_mask(n_k, routes)
    omitted_scores = jnp.where(
        omitted, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype)
    )
    return selected.min(axis=1) - omitted_scores.max(axis=1)


def miss_probability(n_pool: int, u: int, r: int) -> float:
    """Eq 10: P_miss = C(N - u, r) / C(N, r)."""
    if n_pool < 0 or u < 0 or r < 0:
        raise ValueError("N, u, r must be non-negative")
    if r > n_pool:
        raise ValueError("r cannot exceed the residual pool size N")
    if u > n_pool:
        raise ValueError("u cannot exceed the residual pool size N")
    if r > n_pool - u:
        return 0.0
    return math.comb(n_pool - u, r) / math.comb(n_pool, r)


def miss_probability_bound(n_pool: int, u: int, r: int) -> float:
    """Eq 10 upper bound (1 - u/N)^r."""
    if n_pool <= 0:
        raise ValueError("N must be positive")
    return (1.0 - u / n_pool) ** r
