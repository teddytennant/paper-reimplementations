"""Offline fitting and online mapping, the two halves of figure 2.

FULL-HEAD MAPPING is eq 7 support with uniform weights; HEAD-LOCAL is eq 11 support
with uniform weights; CacheBridge is eq 11 support with the ATTN-REPAIR weights of
section 3.3, constructed through FUSED-FIT.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from . import attn_repair, support
from .fused_fit import fused_fit
from .ridge import LAMBDA_DEFAULT, fit_affine
from .rope import apply_rope, invert_rope, rope_angles


@dataclass
class Mapper:
    """The deployed artifact: {W, b} per target layer, head and component, plus support."""
    kind: str                 # "full" or "local"
    layers: jnp.ndarray       # S_l, (Lt, k)
    assign: jnp.ndarray       # a(h), (Ht,)
    w_k: jnp.ndarray          # (Lt, Ht, p, dt)
    b_k: jnp.ndarray          # (Lt, Ht, dt)
    w_v: jnp.ndarray
    b_v: jnp.ndarray

    @property
    def n_coefficients(self) -> int:
        return int(self.w_k.size + self.w_v.size)


def _fit_component(src_z, tgt_z, layers, assign, kind, weights, lam, chunk):
    """One target layer, one component. Returns (W, b) stacked over target KV heads."""
    if kind == "full":
        x = support.full_head_features(src_z, layers)          # (n, k*Hs*ds)
        x = jnp.broadcast_to(x, (tgt_z.shape[1],) + x.shape)   # share the design matrix
        y = jnp.transpose(tgt_z, (1, 0, 2))
        return fit_affine(x, y, weights, lam)
    if chunk is None:
        x = support.head_local_features(src_z, layers, assign)
        y = jnp.transpose(tgt_z, (1, 0, 2))
        return fit_affine(x, y, weights, lam)
    if weights is None:
        weights = jnp.ones((tgt_z.shape[1], tgt_z.shape[0]), tgt_z.dtype)
    return fused_fit(src_z, tgt_z, layers, assign, weights, lam, chunk)


def fit_mapper(src_k, src_v, tgt_k, tgt_v, layers, assign, kind="local",
               weights=None, lam: float = LAMBDA_DEFAULT, chunk=None) -> Mapper:
    """Fit every target layer. src_* are (Ls, n, Hs, ds), tgt_* are (Lt, n, Ht, dt).

    weights, when given, is a pair of (Lt, Ht, n) mean-one weight arrays for K and V.
    """
    wks, bks, wvs, bvs = [], [], [], []
    for l in range(tgt_k.shape[0]):
        wk = None if weights is None else weights[0][l]
        wv = None if weights is None else weights[1][l]
        a, b = _fit_component(src_k, tgt_k[l], layers[l], assign, kind, wk, lam, chunk)
        c, d = _fit_component(src_v, tgt_v[l], layers[l], assign, kind, wv, lam, chunk)
        wks.append(a); bks.append(b); wvs.append(c); bvs.append(d)
    return Mapper(kind, layers, assign,
                  jnp.stack(wks), jnp.stack(bks), jnp.stack(wvs), jnp.stack(bvs))


def _apply_component(src_z, mapper, W, b):
    preds = []
    for l in range(W.shape[0]):
        if mapper.kind == "full":
            x = support.full_head_features(src_z, mapper.layers[l])
            y = jnp.einsum("np,hpd->nhd", x, W[l]) + b[l][None]
        else:
            x = support.head_local_features(src_z, mapper.layers[l], mapper.assign)
            y = jnp.einsum("hnp,hpd->nhd", x, W[l]) + b[l][None]
        preds.append(y)
    return jnp.stack(preds)


def apply_mapper(mapper: Mapper, src_k, src_v):
    """Online mapping in content space. Returns predicted (Lt, n, Ht, dt) K and V."""
    return (_apply_component(src_k, mapper, mapper.w_k, mapper.b_k),
            _apply_component(src_v, mapper, mapper.w_v, mapper.b_v))


def cache_to_content(cache_z, positions, head_dim):
    """Remove source position, the first online step of figure 2."""
    return invert_rope(cache_z, rope_angles(positions, head_dim)[..., None, :])


def content_to_cache(content_z, positions, head_dim):
    """Apply target position, the last online step before cache materialization."""
    return apply_rope(content_z, rope_angles(positions, head_dim)[..., None, :])


def attn_repair_weights(traces, sample_pos, boundaries, q_per_kv, p_local):
    """Build the (Lt, Ht, n) mean-one weight arrays of section 3.3 for K and V."""
    r_k, r_v = attn_repair.boundary_sensitivities(
        traces.tgt_q_rope, traces.tgt_k_rope, traces.tgt_v,
        sample_pos, boundaries, q_per_kv,
    )
    lt, ht = r_k.shape[0], r_k.shape[1]
    r_k = r_k.reshape(lt, ht, -1)
    r_v = r_v.reshape(lt, ht, -1)
    tau = attn_repair.floor_tau(r_k.shape[-1], p_local)
    w_k, alpha_k = attn_repair.shrunk_weights(r_k, tau)
    w_v, alpha_v = attn_repair.shrunk_weights(r_v, tau)
    return (w_k, w_v), (alpha_k, alpha_v), tau
