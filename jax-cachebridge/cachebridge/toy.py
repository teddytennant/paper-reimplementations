"""A tiny synthetic GQA model pair standing in for the Qwen3 and Ministral 3 pairs.

Both models are real grouped-query transformers. The target's content-space K and V
at layer l are planted from the source's at a fixed set of source layers through a
per-head linear map, so the architecture-indexed correspondence a(h) = h of eq 11 is
a fact about the data and the head-assignment ablation of figure 5 has something to
measure. The target then attends to those K and V, so its queries are consistent with
its own cache.

Two deviations keep a single global affine map from being exactly right, which is the
condition under which ATTN-REPAIR can trade coordinate fit for receiver-composed
error: a tanh term, and a position-modulated term that makes the true map drift along
the sequence the way early and late token positions differ in a real cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .rope import apply_rope, rope_angles


@dataclass(frozen=True)
class GQAConfig:
    n_layers: int = 4
    n_kv_heads: int = 4
    q_per_kv: int = 2
    head_dim: int = 16
    vocab: int = 64

    @property
    def n_q_heads(self) -> int:
        return self.n_kv_heads * self.q_per_kv

    @property
    def d_model(self) -> int:
        return self.n_q_heads * self.head_dim


def _rms_norm(x, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x**2, -1, keepdims=True) + eps)


def _normal(key, shape, scale):
    return jax.random.normal(key, shape) * scale


def init_source(key, cfg: GQAConfig):
    keys = jax.random.split(key, 1 + 6 * cfg.n_layers)
    d, hd = cfg.d_model, cfg.head_dim
    p = {"emb": _normal(keys[0], (cfg.vocab, d), 1.0), "layers": []}
    for i in range(cfg.n_layers):
        k = keys[1 + 6 * i: 7 + 6 * i]
        s = d**-0.5
        p["layers"].append({
            "wq": _normal(k[0], (d, cfg.n_q_heads * hd), s),
            "wk": _normal(k[1], (d, cfg.n_kv_heads * hd), s),
            "wv": _normal(k[2], (d, cfg.n_kv_heads * hd), s),
            "wo": _normal(k[3], (cfg.n_q_heads * hd, d), s),
            "w1": _normal(k[4], (d, 2 * d), s),
            "w2": _normal(k[5], (2 * d, d), (2 * d) ** -0.5),
        })
    return p


def init_target(key, cfg: GQAConfig):
    """The target has no K/V projections; its cache comes from the planted relation."""
    keys = jax.random.split(key, 1 + 4 * cfg.n_layers)
    d, hd = cfg.d_model, cfg.head_dim
    p = {"emb": _normal(keys[0], (cfg.vocab, d), 1.0), "layers": []}
    for i in range(cfg.n_layers):
        k = keys[1 + 4 * i: 5 + 4 * i]
        s = d**-0.5
        p["layers"].append({
            "wq": _normal(k[0], (d, cfg.n_q_heads * hd), s),
            "wo": _normal(k[1], (cfg.n_q_heads * hd, d), s),
            "w1": _normal(k[2], (d, 2 * d), s),
            "w2": _normal(k[3], (2 * d, d), (2 * d) ** -0.5),
        })
    return p


def _gqa_attention(q, k, v, cfg: GQAConfig):
    """Causal grouped-query attention. q is (S, T, Hq, hd), k and v are (S, T, H, hd)."""
    s, t = q.shape[0], q.shape[1]
    qg = q.reshape(s, t, cfg.n_kv_heads, cfg.q_per_kv, cfg.head_dim)
    scores = jnp.einsum("sqhgd,skhd->shgqk", qg, k) / jnp.sqrt(float(cfg.head_dim))
    causal = jnp.tril(jnp.ones((t, t), bool))
    scores = jnp.where(causal, scores, -1e30)
    a = jax.nn.softmax(scores, -1)
    o = jnp.einsum("shgqk,skhd->sqhgd", a, v)
    return o.reshape(s, t, cfg.n_q_heads * cfg.head_dim)


def _block_tail(h, layer):
    return h + jax.nn.gelu(_rms_norm(h) @ layer["w1"]) @ layer["w2"]


def run_source(params, cfg: GQAConfig, tokens):
    """Returns content-space K and V per layer, each (Ls, S, T, Hs, ds)."""
    s, t = tokens.shape
    ang = rope_angles(jnp.arange(t), cfg.head_dim)[None, :, None, :]
    h = params["emb"][tokens]
    ks, vs = [], []
    for layer in params["layers"]:
        x = _rms_norm(h)
        q = (x @ layer["wq"]).reshape(s, t, cfg.n_q_heads, cfg.head_dim)
        kc = (x @ layer["wk"]).reshape(s, t, cfg.n_kv_heads, cfg.head_dim)
        vc = (x @ layer["wv"]).reshape(s, t, cfg.n_kv_heads, cfg.head_dim)
        o = _gqa_attention(apply_rope(q, ang), apply_rope(kc, ang), vc, cfg)
        h = _block_tail(h + o @ layer["wo"], layer)
        ks.append(kc)
        vs.append(vc)
    return jnp.stack(ks), jnp.stack(vs)


@dataclass(frozen=True)
class Plant:
    support: tuple            # support[l] is the tuple of source layers feeding target layer l
    mk: jnp.ndarray           # (Lt, Ht, k_true, ds, dt) exact per-head linear part
    mv: jnp.ndarray
    ck: jnp.ndarray           # (Lt, Ht, dt) intercept
    cv: jnp.ndarray
    nk: jnp.ndarray           # (Lt, Ht, ds, dt) tanh term
    nv: jnp.ndarray
    dk: jnp.ndarray           # (Lt, Ht, ds, dt) position-modulated term
    dv: jnp.ndarray
    nonlinear: float
    position_drift: float
    noise: float


def position_phase(seq_len: int):
    """The modulation phi(t) on the drifting part of the planted map, roughly zero mean."""
    return jnp.exp(-jnp.arange(seq_len) / 12.0) - 0.3


def init_plant(key, src_cfg: GQAConfig, tgt_cfg: GQAConfig, k_true: int = 2,
               nonlinear: float = 0.15, position_drift: float = 0.5, noise: float = 0.05):
    support = tuple(
        tuple(sorted(((l + j) % src_cfg.n_layers) for j in range(k_true)))
        for l in range(tgt_cfg.n_layers)
    )
    lt, ht = tgt_cfg.n_layers, tgt_cfg.n_kv_heads
    ds, dt = src_cfg.head_dim, tgt_cfg.head_dim
    keys = jax.random.split(key, 8)
    s = ds**-0.5
    return Plant(
        support=support,
        mk=_normal(keys[0], (lt, ht, k_true, ds, dt), s),
        mv=_normal(keys[1], (lt, ht, k_true, ds, dt), s),
        ck=_normal(keys[2], (lt, ht, dt), 0.1),
        cv=_normal(keys[3], (lt, ht, dt), 0.1),
        nk=_normal(keys[4], (lt, ht, ds, dt), s),
        nv=_normal(keys[5], (lt, ht, ds, dt), s),
        dk=_normal(keys[6], (lt, ht, ds, dt), s),
        dv=_normal(keys[7], (lt, ht, ds, dt), s),
        nonlinear=nonlinear,
        position_drift=position_drift,
        noise=noise,
    )


def _plant_layer(src_z, l, m, c, nmat, dmat, plant: Plant, key, seq_len):
    """One planted target layer.

    sum_j block_j M_{l,h,j} + c + nonlinear tanh(block_0) N + drift phi(t) block_0 D + noise.
    """
    out = 0.0
    for slot, j in enumerate(plant.support[l]):
        out = out + jnp.einsum("sthd,hde->sthe", src_z[j], m[l, :, slot])
    first = src_z[plant.support[l][0]]
    out = out + plant.nonlinear * jnp.einsum("sthd,hde->sthe", jnp.tanh(first), nmat[l])
    phase = position_phase(seq_len)[None, :, None, None]
    out = out + plant.position_drift * phase * jnp.einsum("sthd,hde->sthe", first, dmat[l])
    out = out + c[l]
    return out + plant.noise * jax.random.normal(key, out.shape)


def run_target(params, cfg: GQAConfig, tokens, src_k, src_v, plant: Plant, key):
    """Returns content-space target K and V, post-RoPE queries and post-RoPE keys."""
    s, t = tokens.shape
    ang = rope_angles(jnp.arange(t), cfg.head_dim)[None, :, None, :]
    keys = jax.random.split(key, 2 * cfg.n_layers)
    h = params["emb"][tokens]
    ks, vs, qs, krs = [], [], [], []
    for i, layer in enumerate(params["layers"]):
        kc = _plant_layer(src_k, i, plant.mk, plant.ck, plant.nk, plant.dk, plant,
                          keys[2 * i], t)
        vc = _plant_layer(src_v, i, plant.mv, plant.cv, plant.nv, plant.dv, plant,
                          keys[2 * i + 1], t)
        x = _rms_norm(h)
        q = (x @ layer["wq"]).reshape(s, t, cfg.n_q_heads, cfg.head_dim)
        qr, kr = apply_rope(q, ang), apply_rope(kc, ang)
        h = _block_tail(h + _gqa_attention(qr, kr, vc, cfg) @ layer["wo"], layer)
        ks.append(kc)
        vs.append(vc)
        qs.append(qr)
        krs.append(kr)
    return jnp.stack(ks), jnp.stack(vs), jnp.stack(qs), jnp.stack(krs)


@dataclass(frozen=True)
class Traces:
    """Content-space calibration traces plus the receiver state ATTN-REPAIR needs."""
    src_k: jnp.ndarray   # (Ls, S, T, Hs, ds)
    src_v: jnp.ndarray
    tgt_k: jnp.ndarray   # (Lt, S, T, Ht, dt)
    tgt_v: jnp.ndarray
    tgt_q_rope: jnp.ndarray   # (Lt, S, T, Hq, dt)
    tgt_k_rope: jnp.ndarray
    tokens: jnp.ndarray


def make_traces(key, src_cfg: GQAConfig, tgt_cfg: GQAConfig, plant: Plant,
                n_seq: int, seq_len: int, src_params, tgt_params):
    tokens = jax.random.randint(key, (n_seq, seq_len), 0, src_cfg.vocab)
    src_k, src_v = run_source(src_params, src_cfg, tokens)
    tgt_k, tgt_v, q, kr = run_target(tgt_params, tgt_cfg, tokens, src_k, src_v, plant,
                                     jax.random.fold_in(key, 7))
    return Traces(src_k, src_v, tgt_k, tgt_v, q, kr, tokens)


def flatten_positions(traces: Traces, sample_pos):
    """Stack (sequence, sampled position) into the n calibration rows the fitter sees."""
    def flat(z):
        sel = jnp.take(z, sample_pos, axis=2)
        return sel.reshape(sel.shape[0], -1, sel.shape[3], sel.shape[4])

    return flat(traces.src_k), flat(traces.src_v), flat(traces.tgt_k), flat(traces.tgt_v)
