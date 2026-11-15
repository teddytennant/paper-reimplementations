"""Visual-token scores for DIPrune.

Text tokens are never scored. Ranks use 1 for the most salient visual token.
The paper leaves rank ties and the query-window rounding rule unspecified.
Ties break toward the smaller index. The window length uses Python round.
"""

import jax.numpy as jnp
import numpy as np


def aggregate_attention(attention):
    """Mean text-to-visual attention over queries, after heads if present.

    ``attention`` is ``[M, N]`` or ``[H, M, N]``, already the text-query by
    visual-key weights. If it is 3D, heads are averaged first, then queries.
    This is Eq. (raw_importance) / (cross_modal_attention) in the methodology.
    """
    attention = jnp.asarray(attention)
    if attention.ndim == 3:
        attention = jnp.mean(attention, axis=0)
    if attention.ndim != 2:
        raise ValueError("attention must have shape (M, N) or (H, M, N)")
    if attention.shape[0] == 0 or attention.shape[1] == 0:
        raise ValueError("attention must have at least one query and one key")
    return jnp.mean(attention, axis=0)


def window_length(n_queries, window_ratio=0.8):
    """Recent-window length, ``max(1, min(M, round(window_ratio * M)))``.

    The appendix sets the ratio to 0.8 and writes ``L_win`` as about 0.8 times
    the text length. It does not state a rounding rule. ``round`` is Python
    ``round``, which sends halves to even, then the result is clamped to
    ``[1, M]``.
    """
    if isinstance(n_queries, bool) or not isinstance(n_queries, (int, np.integer)):
        raise ValueError("n_queries must be an integer")
    n_queries = int(n_queries)
    if n_queries < 1:
        raise ValueError("n_queries must be at least 1")
    try:
        ratio = float(window_ratio)
    except TypeError as exc:
        raise ValueError("window_ratio must be a real number") from exc
    if ratio < 0:
        raise ValueError("window_ratio must be non-negative")
    return max(1, min(n_queries, round(ratio * n_queries)))


def decoupled_attention(queries, visual_keys, n_heads, window_ratio=0.8):
    """Saliency from a recent text-query window against visual keys.

    ``queries`` is ``[M, D]`` and ``visual_keys`` is ``[N, D]``. ``D`` must be
    divisible by ``n_heads``. Heads are contiguous blocks of width
    ``d_h = D / n_heads``. The window is the last ``L_win`` rows (the appendix
    calls this a recent text query window and does not write the slice).

    For each head, softmax is over the visual key axis of
    ``Q_sub^{(h)} (K_vis^{(h)})^T / sqrt(d_h)``. Head scores are averaged,
    then averaged over the window queries.
    """
    queries = jnp.asarray(queries)
    visual_keys = jnp.asarray(visual_keys)
    if queries.ndim != 2 or visual_keys.ndim != 2:
        raise ValueError("queries and visual_keys must have shape (length, D)")
    if queries.shape[1] != visual_keys.shape[1]:
        raise ValueError("queries and visual_keys must share D")
    if queries.shape[0] < 1 or visual_keys.shape[0] < 1:
        raise ValueError("queries and visual_keys must be non-empty")
    if isinstance(n_heads, bool) or not isinstance(n_heads, (int, np.integer)):
        raise ValueError("n_heads must be an integer")
    n_heads = int(n_heads)
    if n_heads < 1:
        raise ValueError("n_heads must be at least 1")
    dim = int(queries.shape[1])
    if dim % n_heads != 0:
        raise ValueError("D must be divisible by n_heads")

    length = window_length(int(queries.shape[0]), window_ratio)
    q_sub = queries[-length:]
    d_h = dim // n_heads
    q_heads = q_sub.reshape(length, n_heads, d_h).transpose(1, 0, 2)
    k_heads = visual_keys.reshape(visual_keys.shape[0], n_heads, d_h).transpose(1, 0, 2)
    scale = jnp.sqrt(jnp.asarray(d_h, dtype=q_heads.dtype))
    logits = jnp.matmul(q_heads, jnp.swapaxes(k_heads, -1, -2)) / scale
    attention = _softmax(logits)
    averaged = jnp.mean(attention, axis=0)
    return jnp.mean(averaged, axis=0)


def _softmax(logits):
    """Softmax over the last axis, which is the visual key axis."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    return weights / jnp.sum(weights, axis=-1, keepdims=True)


def _largest_first(scores):
    """Indices sorted by score descending, then by smaller index."""
    scores = jnp.asarray(scores)
    index = jnp.arange(scores.shape[0])
    return jnp.lexsort((index, -scores))


def ranks_from_scores(scores):
    """Rank in ``{1, ..., N}``. Rank 1 is the largest score.

    Ties are not specified in the paper. The smaller index gets the better
    (smaller) rank, via a stable sort: lexicographic order on
    ``(-score, index)``.
    """
    scores = jnp.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D vector over visual tokens")
    n = int(scores.shape[0])
    if n < 1:
        raise ValueError("scores must be non-empty")
    order = _largest_first(scores)
    ranks = jnp.zeros(n, dtype=jnp.int32)
    return ranks.at[order].set(jnp.arange(1, n + 1, dtype=jnp.int32))


def alignment_score(ranks):
    """Inverted rank normalization, ``S_align = (N - R + 1) / N``."""
    ranks = jnp.asarray(ranks)
    if ranks.ndim != 1:
        raise ValueError("ranks must be a 1D vector")
    n = int(ranks.shape[0])
    if n < 1:
        raise ValueError("ranks must be non-empty")
    dtype = jnp.float64 if jnp.asarray(1.0).dtype == jnp.float64 else jnp.float32
    ranks = ranks.astype(dtype)
    width = jnp.asarray(n, dtype=dtype)
    return (width - ranks + 1) / width


def semantic_alignment(scores):
    """``S_align`` from raw visual saliency. Scale-invariant for positive scales."""
    return alignment_score(ranks_from_scores(scores))


def semantic_evolution(ranks_prev, ranks_curr, m_max=200):
    """Bounded rank displacement between two caller-chosen layers.

    ``Delta R = R^{(l-delta)} - R^{(l)}``. A positive value means the token
    rose in priority (its rank number fell). The paper default delta is 1 and
    the paper default ``M_max`` is 200. Neither a layer index nor delta is
    hardcoded here. The caller passes the two rank vectors.

    ``S_evol = 0.5 * (clip(Delta R, -M_max, M_max) / M_max + 1)``.
    """
    ranks_prev = jnp.asarray(ranks_prev)
    ranks_curr = jnp.asarray(ranks_curr)
    if ranks_prev.shape != ranks_curr.shape or ranks_prev.ndim != 1:
        raise ValueError("ranks_prev and ranks_curr must be 1D vectors of equal length")
    if ranks_prev.shape[0] < 1:
        raise ValueError("ranks must be non-empty")
    try:
        limit = float(m_max)
    except TypeError as exc:
        raise ValueError("m_max must be a positive real number") from exc
    if limit <= 0:
        raise ValueError("m_max must be positive")
    dtype = jnp.float64 if jnp.asarray(1.0).dtype == jnp.float64 else jnp.float32
    delta = ranks_prev.astype(dtype) - ranks_curr.astype(dtype)
    clipped = jnp.clip(delta, -limit, limit)
    return 0.5 * (clipped / jnp.asarray(limit, dtype=dtype) + 1)


def fuse_scores(s_align, s_evol, alpha=0.6, beta=0.4):
    """``S = alpha * S_align + beta * S_evol``. Requires ``alpha + beta == 1``.

    The paper defaults are ``alpha = 0.6`` and ``beta = 0.4``. The sum must
    match 1 within ``1e-6``, otherwise ``ValueError``.
    """
    try:
        total = float(alpha) + float(beta)
    except TypeError as exc:
        raise ValueError("alpha and beta must be real numbers") from exc
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"alpha + beta must equal 1 within 1e-6, got {total}")
    s_align = jnp.asarray(s_align)
    s_evol = jnp.asarray(s_evol)
    if s_align.shape != s_evol.shape:
        raise ValueError("s_align and s_evol must share a shape")
    return alpha * s_align + beta * s_evol
