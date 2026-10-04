"""Fixed-stride segment aggregation with separate paper and air pools.

S = ceil(T / gamma). The last window is zero-padded and those padded steps
are masked out, so they are neither paper nor air. Paper points and air
points inside a window are averaged separately and never mixed into one mean.
"""

from __future__ import annotations

import jax.numpy as jnp


def n_segments(length: int, gamma: int) -> int:
    if gamma < 1:
        raise ValueError("gamma must be positive")
    if length < 0:
        raise ValueError("length must be non-negative")
    return (int(length) + int(gamma) - 1) // int(gamma)


def _as_batched(h, q):
    if h.ndim == 2:
        return h[None, ...], q[None, ...], True
    if h.ndim != 3:
        raise ValueError("h must have shape (T, d) or (batch, T, d)")
    return h, q, False


def _squeeze_if(tree, unbatched):
    if not unbatched:
        return tree
    out = {}
    for key, value in tree.items():
        out[key] = value[0]
    return out


def aggregate_segments(h, q, gamma: int):
    """Pool each fixed window into separate paper and air statistics.

    Parameters
    ----------
    h:
        Hidden sequence, shape (batch, T, d) or (T, d).
    q:
        Paper mask, 1 on paper and 0 in air. Same time layout as h.
    gamma:
        Segment length. Counts in the token features are divided by gamma,
        not by the number of valid steps.

    Returns
    -------
    dict with paper_mean, air_mean, n_paper, n_air, rho, paper_dominant,
    features, and valid_mask. features is the concatenation
    [paper_mean, air_mean, n_paper / gamma, n_air / gamma].
    """
    h, q, unbatched = _as_batched(h, jnp.asarray(q))
    h = jnp.asarray(h)
    q = jnp.asarray(q, dtype=h.dtype)
    if q.shape != h.shape[:2]:
        raise ValueError("q must match the time shape of h")
    gamma = int(gamma)
    batch, length, dim = h.shape
    n_seg = n_segments(length, gamma) if length > 0 else 0
    if n_seg == 0:
        empty = {
            "paper_mean": jnp.zeros((batch, 0, dim), dtype=h.dtype),
            "air_mean": jnp.zeros((batch, 0, dim), dtype=h.dtype),
            "n_paper": jnp.zeros((batch, 0), dtype=h.dtype),
            "n_air": jnp.zeros((batch, 0), dtype=h.dtype),
            "rho": jnp.zeros((batch, 0), dtype=h.dtype),
            "paper_dominant": jnp.zeros((batch, 0), dtype=jnp.bool_),
            "features": jnp.zeros((batch, 0, 2 * dim + 2), dtype=h.dtype),
            "valid_mask": jnp.zeros((batch, 0, gamma), dtype=h.dtype),
        }
        return _squeeze_if(empty, unbatched)

    pad = n_seg * gamma - length
    h_pad = jnp.pad(h, ((0, 0), (0, pad), (0, 0)))
    q_pad = jnp.pad(q, ((0, 0), (0, pad)))
    valid = jnp.pad(jnp.ones((batch, length), dtype=h.dtype), ((0, 0), (0, pad)))

    h_seg = h_pad.reshape(batch, n_seg, gamma, dim)
    q_seg = q_pad.reshape(batch, n_seg, gamma)
    m_seg = valid.reshape(batch, n_seg, gamma)

    paper_w = q_seg * m_seg
    air_w = (1.0 - q_seg) * m_seg
    n_paper = jnp.sum(paper_w, axis=-1)
    n_air = jnp.sum(air_w, axis=-1)

    sum_paper = jnp.sum(paper_w[..., None] * h_seg, axis=2)
    sum_air = jnp.sum(air_w[..., None] * h_seg, axis=2)
    paper_mean = sum_paper / jnp.maximum(n_paper, 1.0)[..., None]
    air_mean = sum_air / jnp.maximum(n_air, 1.0)[..., None]
    paper_mean = jnp.where((n_paper > 0)[..., None], paper_mean, 0.0)
    air_mean = jnp.where((n_air > 0)[..., None], air_mean, 0.0)

    rho = n_paper / jnp.maximum(n_paper + n_air, 1.0)
    paper_dominant = rho >= 0.5
    features = segment_features(paper_mean, air_mean, n_paper, n_air, gamma)
    packed = {
        "paper_mean": paper_mean,
        "air_mean": air_mean,
        "n_paper": n_paper,
        "n_air": n_air,
        "rho": rho,
        "paper_dominant": paper_dominant,
        "features": features,
        "valid_mask": m_seg,
    }
    return _squeeze_if(packed, unbatched)


def segment_features(paper_mean, air_mean, n_paper, n_air, gamma: int):
    """Build the linear-map input, with counts normalized by gamma."""
    gamma_f = jnp.asarray(gamma, dtype=paper_mean.dtype)
    counts = jnp.stack([n_paper / gamma_f, n_air / gamma_f], axis=-1)
    return jnp.concatenate([paper_mean, air_mean, counts], axis=-1)


def project_tokens(features, weight, bias):
    """Map segment features in R^{2d+2} to tokens in R^d."""
    return features @ weight.T + bias
