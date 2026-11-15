"""BSCA scoring: time/relation shifts, complex attention, Hamilton composition."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bsca.biquaternion import bq_inner, hamilton

ParamTree = dict


def _complex_normal(key, shape, scale=0.1, dtype=jnp.complex64):
    k1, k2 = jax.random.split(key)
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    re = scale * jax.random.normal(k1, shape, dtype=real_dtype)
    im = scale * jax.random.normal(k2, shape, dtype=real_dtype)
    return jax.lax.complex(re, im).astype(dtype)


def init_params(key, n_entities, n_relations, n_timestamps, k, scale=0.1):
    """Random complex embeddings.

    Relation tensors have length 2 * n_relations: the second half is r^{-1}.
    Timestamp tensors are (n_timestamps, 4, k) for ▲, ▼, and ♦.
    """
    keys = jax.random.split(key, 8)
    shape_e = (n_entities, 4, k)
    shape_r = (2 * n_relations, 4, k)
    shape_t = (n_timestamps, 4, k)
    params = {
        "entities": _complex_normal(keys[0], shape_e, scale),
        "rel_mul": _complex_normal(keys[1], shape_r, scale),
        "rel_add": _complex_normal(keys[2], shape_r, scale),
        "time_mul": _complex_normal(keys[3], shape_t, scale),
        "time_add": _complex_normal(keys[4], shape_t, scale),
        "time_joint": _complex_normal(keys[5], shape_t, scale),
        "attn": _complex_normal(keys[6], (4, k), scale),
    }
    return params


def n_relations(params):
    return int(params["rel_mul"].shape[0] // 2)


def fuse_attention(a, e_time, e_rel):
    """Complex-valued attention (eq. 10).

    Scores are Re of the Hermitian inner product with a, then softmax.
    Returns fused embedding and real weights (a_tau, a_rel).
    """
    alpha_tau = bq_inner(a, e_time)
    alpha_rel = bq_inner(a, e_rel)
    logits = jnp.stack([alpha_tau, alpha_rel], axis=-1)
    weights = jax.nn.softmax(logits.real if jnp.iscomplexobj(logits) else logits, axis=-1)
    a_tau = weights[..., 0]
    a_rel = weights[..., 1]
    fused = a_tau[..., None, None] * e_time + a_rel[..., None, None] * e_rel
    return fused, (a_tau, a_rel)


def _gather(table, idx):
    idx = jnp.asarray(idx)
    return table[idx]


def entity_shifts(params, e, r, tau):
    """eq. 8: e_τ = e + e_τ▼, e_r = e + e_r▼."""
    e_tau = e + _gather(params["time_add"], tau)
    e_rel = e + _gather(params["rel_add"], r)
    return e_tau, e_rel


def relation_time(params, r, tau):
    """eq. 9: e_rτ = e_r▲ + e_τ♦, e_rττ = Hamilton(e_rτ, e_τ▲)."""
    e_r_tau = _gather(params["rel_mul"], r) + _gather(params["time_joint"], tau)
    e_rtt = hamilton(e_r_tau, _gather(params["time_mul"], tau))
    return e_r_tau, e_rtt


def encode(params, s, r, tau):
    """Subject-side encoding: e_s^{rττ} = Hamilton(e_sτr, e_rττ)."""
    e_s = _gather(params["entities"], s)
    e_s_tau, e_s_r = entity_shifts(params, e_s, r, tau)
    e_str, attn = fuse_attention(params["attn"], e_s_tau, e_s_r)
    _, e_rtt = relation_time(params, r, tau)
    e_hat = hamilton(e_str, e_rtt)
    return e_hat, e_str, e_rtt, attn


def object_fusion(params, o, r, tau):
    """Object-side fusion e_oτr with scores from e_oτ, e_or (for N3)."""
    e_o = _gather(params["entities"], o)
    e_o_tau, e_o_r = entity_shifts(params, e_o, r, tau)
    e_otr, attn = fuse_attention(params["attn"], e_o_tau, e_o_r)
    return e_otr, attn


def score(params, s, r, o, tau):
    """φ(s, r, o, τ) = ⟨e_s^{rττ}, e_o⟩, a real scalar (or batch of scalars)."""
    e_hat, _, _, _ = encode(params, s, r, tau)
    e_o = _gather(params["entities"], o)
    return bq_inner(e_hat, e_o)


def scores_all_objects(params, s, r, tau):
    """Scores against every entity as object. Returns (..., n_entities), plus factors."""
    e_hat, e_str, e_rtt, attn = encode(params, s, r, tau)
    ents = params["entities"]
    scores = jnp.real(jnp.einsum("...ck,nck->...n", jnp.conj(e_hat), ents))
    return scores, e_str, e_rtt, attn


def filtered_ranks(params, s, r, o, tau, facts=None):
    """1-based filtered object ranks. Other true tails at (s, r, τ) are masked."""
    s = jnp.asarray(s)
    r = jnp.asarray(r)
    o = jnp.asarray(o)
    tau = jnp.asarray(tau)
    squeeze = s.ndim == 0
    if squeeze:
        s, r, o, tau = jnp.expand_dims(s, 0), jnp.expand_dims(r, 0), jnp.expand_dims(o, 0), jnp.expand_dims(tau, 0)

    scores, _, _, _ = scores_all_objects(params, s, r, tau)
    n_ent = scores.shape[-1]
    if facts is not None:
        facts = jnp.asarray(facts)
        if facts.ndim == 1:
            facts = facts[None, :]
        ks, kr, ko, kt = facts[:, 0], facts[:, 1], facts[:, 2], facts[:, 3]
        match = (
            (ks[None, :] == s[:, None])
            & (kr[None, :] == r[:, None])
            & (kt[None, :] == tau[:, None])
            & (ko[None, :] != o[:, None])
        )
        dummy = n_ent
        idx = jnp.where(match, ko[None, :], dummy)
        pad = jnp.concatenate([scores, jnp.full((scores.shape[0], 1), 0, dtype=scores.dtype)], axis=-1)
        batch_ix = jnp.arange(scores.shape[0])[:, None]
        pad = pad.at[batch_ix, idx].set(jnp.array(-1.0e9, dtype=scores.dtype))
        scores = pad[:, :n_ent]

    true_scores = jnp.take_along_axis(scores, o[:, None], axis=-1)[:, 0]
    ranks = 1 + jnp.sum(scores > true_scores[:, None], axis=-1)
    if squeeze:
        return ranks[0]
    return ranks
