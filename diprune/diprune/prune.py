"""Keep the K highest-scoring visual tokens. Text tokens are never removed."""

import jax.numpy as jnp
import numpy as np


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def select_topk(scores, k):
    """Indices of the K largest scores, in original relative order.

    Ties are not specified in the paper. The smaller index is preferred for
    inclusion. ``k`` must lie in ``1..N``.
    """
    scores = jnp.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D vector over visual tokens")
    n = int(scores.shape[0])
    if n < 1:
        raise ValueError("scores must be non-empty")
    k = _as_int(k, "k")
    if k < 1 or k > n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    index = jnp.arange(n)
    order = jnp.lexsort((index, -scores))
    return jnp.sort(order[:k])


def prune_tokens(tokens, visual_mask, scores, k):
    """Drop unselected visual tokens and keep every non-visual token.

    ``tokens`` has shape ``[S, ...]``. ``visual_mask`` is a boolean vector of
    length ``S``. ``scores`` has one entry per visual token, in the order those
    tokens appear. The K selected visual tokens and all text tokens are
    returned in their original relative order, along with the kept indices
    into the original sequence.
    """
    tokens = jnp.asarray(tokens)
    visual_mask = jnp.asarray(visual_mask)
    scores = jnp.asarray(scores)
    if visual_mask.ndim != 1 or visual_mask.shape[0] != tokens.shape[0]:
        raise ValueError("visual_mask must be a vector matching the token sequence")
    if tokens.shape[0] < 1:
        raise ValueError("tokens must be non-empty")
    mask = visual_mask.astype(bool)
    visual_positions = jnp.where(mask)[0]
    n_visual = int(visual_positions.shape[0])
    if scores.ndim != 1 or int(scores.shape[0]) != n_visual:
        raise ValueError("scores length must equal the number of visual tokens")
    local = select_topk(scores, k)
    kept_visual = visual_positions[local]
    text_positions = jnp.where(~mask)[0]
    kept = jnp.sort(jnp.concatenate([kept_visual, text_positions]))
    return tokens[kept], kept
