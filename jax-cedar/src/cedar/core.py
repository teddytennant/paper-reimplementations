"""CEDAR primitives from arXiv:2609.07237 (equations 1-11)."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.nn import logsumexp

Array = jax.Array


def scale_tau(dim: int, dtype) -> Array:
    """Attention temperature tau = sqrt(d) as in eq 1."""
    return jnp.sqrt(jnp.asarray(dim, dtype=dtype))


def token_chunk_ids(boundaries: Array, n_tokens: int) -> Array:
    """Chunk index for each token given exclusive-end boundaries of shape [C+1]."""
    return jnp.searchsorted(boundaries[1:], jnp.arange(n_tokens), side="right")


def chunk_summaries(
    k: Array,
    v: Array,
    boundaries: Array,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Mean key/value summaries and max L2 dispersions (eq 3, 7, 8).

    Args:
        k: [T, D] keys.
        v: [T, D] values.
        boundaries: [C+1] increasing ints, chunk c is [b_c, b_{c+1}).

    Returns:
        kbar: [C, D], vbar: [C, D], n: [C] int lengths,
        r_k: [C] max_j ||k_j - kbar_c||_2,
        r_v: [C] max_j ||v_j - vbar_c||_2.
    """
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    b = jnp.asarray(boundaries)
    starts = b[:-1]
    ends = b[1:]
    n = ends - starts
    t = jnp.arange(k.shape[0])
    in_chunk = (t[None, :] >= starts[:, None]) & (t[None, :] < ends[:, None])
    n_f = n.astype(k.dtype)
    mask = in_chunk[:, :, None].astype(k.dtype)
    kbar = (mask * k[None, :, :]).sum(axis=1) / n_f[:, None]
    vbar = (mask * v[None, :, :]).sum(axis=1) / n_f[:, None]
    kd = k[None, :, :] - kbar[:, None, :]
    vd = v[None, :, :] - vbar[:, None, :]
    kn = jnp.sqrt(jnp.sum(kd * kd, axis=-1))
    vn = jnp.sqrt(jnp.sum(vd * vd, axis=-1))
    r_k = jnp.max(jnp.where(in_chunk, kn, jnp.zeros_like(kn)), axis=-1)
    r_v = jnp.max(jnp.where(in_chunk, vn, jnp.zeros_like(vn)), axis=-1)
    return kbar, vbar, n, r_k, r_v


def _query_positions(
    q: Array,
    k: Array,
    causal: bool,
    query_positions: Optional[Array],
) -> Optional[Array]:
    if not causal:
        return None
    if query_positions is not None:
        return jnp.asarray(query_positions)
    if q.shape[0] != k.shape[0]:
        raise ValueError("query_positions is required when causal=True and Q != T")
    return jnp.arange(q.shape[0])


def _broadcast_refined(refined: Array, n_queries: int, n_chunks: int) -> Array:
    refined = jnp.asarray(refined, dtype=bool)
    if refined.ndim == 1:
        refined = jnp.broadcast_to(refined[None, :], (n_queries, n_chunks))
    return refined


def dense_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    tau: Optional[Array] = None,
    causal: bool = False,
    query_positions: Optional[Array] = None,
) -> Array:
    """Dense causal/non-causal softmax attention (eq 1)."""
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    if tau is None:
        tau = scale_tau(q.shape[-1], q.dtype)
    logits = (q @ k.T) / tau
    qpos = _query_positions(q, k, causal, query_positions)
    if qpos is not None:
        key_pos = jnp.arange(k.shape[0])
        logits = jnp.where(
            key_pos[None, :] <= qpos[:, None],
            logits,
            jnp.full((), -jnp.inf, dtype=logits.dtype),
        )
    weights = jnp.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    return weights @ v


def _hybrid_weights(
    q: Array,
    k: Array,
    v: Array,
    boundaries: Array,
    refined: Array,
    *,
    tau: Optional[Array],
    causal: bool,
    query_positions: Optional[Array],
    summaries: Optional[Tuple[Array, Array, Array, Array, Array]],
    include_summaries: bool,
    double_count: bool,
) -> Tuple[Array, Array, Array, Array]:
    """Token and summary unnormalized weights for hybrid attention (eq 6)."""
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    b = jnp.asarray(boundaries)
    if tau is None:
        tau = scale_tau(q.shape[-1], q.dtype)
    if summaries is None:
        kbar, vbar, n, r_k, r_v = chunk_summaries(k, v, b)
    else:
        kbar, vbar, n, r_k, r_v = summaries
    n_queries = q.shape[0]
    n_chunks = kbar.shape[0]
    refined = _broadcast_refined(refined, n_queries, n_chunks)
    chunk_of_tok = token_chunk_ids(b, k.shape[0])
    use_token = refined[:, chunk_of_tok]
    use_summary = jnp.logical_not(refined) if include_summaries else jnp.zeros_like(refined)
    if double_count:
        use_summary = jnp.ones_like(refined) if include_summaries else use_summary
    qpos = _query_positions(q, k, causal, query_positions)
    if qpos is not None:
        key_pos = jnp.arange(k.shape[0])
        use_token = use_token & (key_pos[None, :] <= qpos[:, None])
        ends = b[1:]
        use_summary = use_summary & (ends[None, :] <= qpos[:, None])
    ninf = jnp.full((), -jnp.inf, dtype=q.dtype)
    tok_logits = jnp.where(use_token, (q @ k.T) / tau, ninf)
    ell = (q @ kbar.T) / tau + jnp.log(n.astype(q.dtype))
    sum_logits = jnp.where(use_summary, ell, ninf)
    return tok_logits, sum_logits, v, vbar


def hybrid_attention(
    q: Array,
    k: Array,
    v: Array,
    boundaries: Array,
    refined: Array,
    *,
    tau: Optional[Array] = None,
    causal: bool = False,
    query_positions: Optional[Array] = None,
    summaries: Optional[Tuple[Array, Array, Array, Array, Array]] = None,
    double_count: bool = False,
) -> Array:
    """Hybrid exact-summary attention (eq 6).

    Refined chunks contribute only their exact tokens. Unrefined chunks
    contribute only length-corrected summaries. Set double_count=True to
    reproduce the incorrect path that also keeps summaries of refined chunks.
    """
    tok_logits, sum_logits, v, vbar = _hybrid_weights(
        q,
        k,
        v,
        boundaries,
        refined,
        tau=tau,
        causal=causal,
        query_positions=query_positions,
        summaries=summaries,
        include_summaries=True,
        double_count=double_count,
    )
    m_tok = jnp.max(tok_logits, axis=-1)
    m_sum = jnp.max(sum_logits, axis=-1)
    m = jnp.maximum(m_tok, m_sum)
    tok_w = jnp.exp(tok_logits - m[:, None])
    sum_w = jnp.exp(sum_logits - m[:, None])
    num = tok_w @ v + sum_w @ vbar
    den = tok_w.sum(axis=-1) + sum_w.sum(axis=-1)
    return num / den[:, None]


def hard_sparse_attention(
    q: Array,
    k: Array,
    v: Array,
    boundaries: Array,
    refined: Array,
    *,
    tau: Optional[Array] = None,
    causal: bool = False,
    query_positions: Optional[Array] = None,
) -> Array:
    """Hard sparse attention over exact tokens only (eq 2)."""
    tok_logits, _, v, _ = _hybrid_weights(
        q,
        k,
        v,
        boundaries,
        refined,
        tau=tau,
        causal=causal,
        query_positions=query_positions,
        summaries=None,
        include_summaries=False,
        double_count=False,
    )
    weights = jnp.exp(tok_logits - logsumexp(tok_logits, axis=-1, keepdims=True))
    return weights @ v


def summary_logits(
    q: Array,
    kbar: Array,
    n: Array,
    tau: Array,
) -> Array:
    """Length-corrected summary logits ell_ic^sum (eq 4)."""
    return (q @ kbar.T) / tau + jnp.log(n.astype(q.dtype))


def refinement_priority(
    q: Array,
    kbar: Array,
    vbar: Array,
    n: Array,
    r_k: Array,
    r_v: Array,
    *,
    tau: Optional[Array] = None,
    rho: Optional[Array] = None,
) -> Array:
    """Error-aware refinement priority pi_ic (eq 10)."""
    q = jnp.asarray(q)
    kbar = jnp.asarray(kbar)
    vbar = jnp.asarray(vbar)
    if tau is None:
        tau = scale_tau(q.shape[-1], q.dtype)
    ell = summary_logits(q, kbar, n, tau)
    qn = jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True))
    eps = (qn * r_k[None, :]) / tau
    vnorm = jnp.sqrt(jnp.sum(vbar * vbar, axis=-1))
    if rho is None:
        rho_i = jnp.max(vnorm)
    else:
        rho_arr = jnp.asarray(rho, dtype=q.dtype)
        rho_i = rho_arr if rho_arr.ndim == 0 else rho_arr[:, None]
    bracket = (jnp.exp(eps) - 1.0) * (vnorm[None, :] + rho_i) + jnp.exp(eps) * r_v[None, :]
    return jnp.exp(ell) * bracket


def greedy_refine(
    priority: Array,
    lengths: Array,
    budget: int,
    *,
    force: Optional[Array] = None,
    unit: str = "chunks",
) -> Array:
    """Greedy chunk selection under a chunk or token budget.

    Forced chunks are always refined. Remaining budget is filled from
    highest priority (eq 10) downward.
    """
    priority = jnp.asarray(priority)
    lengths = jnp.asarray(lengths)
    n_queries, n_chunks = priority.shape
    if force is None:
        force = jnp.zeros((n_queries, n_chunks), dtype=bool)
    else:
        force = _broadcast_refined(force, n_queries, n_chunks)
    if unit == "chunks":
        n_force = force.sum(axis=-1, keepdims=True)
        remain = jnp.maximum(jnp.asarray(budget) - n_force, 0)
        pri = jnp.where(force, jnp.full((), -jnp.inf, dtype=priority.dtype), priority)
        order = jnp.argsort(-pri, axis=-1)
        ranks = jnp.argsort(order, axis=-1)
        extra = ranks < remain
        return force | extra
    if unit == "tokens":
        force_tokens = (force.astype(lengths.dtype) * lengths[None, :]).sum(
            axis=-1, keepdims=True
        )
        remain = jnp.maximum(jnp.asarray(budget) - force_tokens, 0)
        pri = jnp.where(force, jnp.full((), -jnp.inf, dtype=priority.dtype), priority)
        order = jnp.argsort(-pri, axis=-1)
        slen = lengths[order]
        csum = jnp.cumsum(slen, axis=-1)
        sel_sorted = csum <= remain
        extra = jnp.zeros_like(force)
        q_idx = jnp.broadcast_to(jnp.arange(n_queries)[:, None], order.shape)
        extra = extra.at[q_idx, order].set(sel_sorted)
        return force | extra
    raise ValueError("unit must be 'chunks' or 'tokens'")


def forced_refinement_mask(
    boundaries: Array,
    query_positions: Array,
    *,
    sink_chunks: int = 0,
    window_chunks: int = 0,
) -> Array:
    """Causal forced set: current chunk, optional sink prefix, recent window."""
    b = jnp.asarray(boundaries)
    qpos = jnp.asarray(query_positions)
    n_chunks = b.shape[0] - 1
    current = jnp.searchsorted(b[1:], qpos, side="right")
    c_idx = jnp.arange(n_chunks)
    mask = c_idx[None, :] == current[:, None]
    if sink_chunks > 0:
        mask = mask | (c_idx[None, :] < sink_chunks)
    if window_chunks > 0:
        mask = mask | ((c_idx[None, :] < current[:, None]) & (c_idx[None, :] >= current[:, None] - window_chunks))
    return mask


def reconstruction_bound(
    q: Array,
    k: Array,
    v: Array,
    boundaries: Array,
    refined: Array,
    *,
    tau: Optional[Array] = None,
    causal: bool = False,
    query_positions: Optional[Array] = None,
    M: Optional[Array] = None,
    summaries: Optional[Tuple[Array, Array, Array, Array, Array]] = None,
) -> Array:
    """Proposition 1 output-error bound (eq 11), shape [Q]."""
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    b = jnp.asarray(boundaries)
    if tau is None:
        tau = scale_tau(q.shape[-1], q.dtype)
    if summaries is None:
        kbar, vbar, n, r_k, r_v = chunk_summaries(k, v, b)
    else:
        kbar, vbar, n, r_k, r_v = summaries
    n_queries = q.shape[0]
    refined = _broadcast_refined(refined, n_queries, kbar.shape[0])
    unrefined = jnp.logical_not(refined)
    qpos = _query_positions(q, k, causal, query_positions)
    tok_logits = (q @ k.T) / tau
    if qpos is not None:
        key_pos = jnp.arange(k.shape[0])
        tok_logits = jnp.where(
            key_pos[None, :] <= qpos[:, None],
            tok_logits,
            jnp.full((), -jnp.inf, dtype=tok_logits.dtype),
        )
        ends = b[1:]
        unrefined = unrefined & (ends[None, :] <= qpos[:, None])
    log_Z = logsumexp(tok_logits, axis=-1)
    zbar = (q @ kbar.T) / tau
    qn = jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True))
    eps = (qn * r_k[None, :]) / tau
    vnorm = jnp.sqrt(jnp.sum(vbar * vbar, axis=-1))
    if M is None:
        vn_tok = jnp.sqrt(jnp.sum(v * v, axis=-1))
        M_i = jnp.max(vn_tok)
    else:
        M_arr = jnp.asarray(M, dtype=q.dtype)
        M_i = M_arr if M_arr.ndim == 0 else M_arr[:, None]
    bracket = jnp.exp(eps) * r_v[None, :] + (jnp.exp(eps) - 1.0) * (vnorm[None, :] + M_i)
    mass = n.astype(q.dtype) * jnp.exp(zbar - log_Z[:, None])
    return jnp.sum(jnp.where(unrefined, mass * bracket, 0), axis=-1)
