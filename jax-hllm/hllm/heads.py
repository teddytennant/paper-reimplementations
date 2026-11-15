"""The three scoring heads of Figure 2.

Each head takes the per-item hidden states h_1..h_N in R^{N x D} that a single
prefill pass leaves at the readout token of every candidate, and emits the
N x K item-position score matrix M that equations (1) to (3) consume. The
backbone that produces those hidden states is not in this repository.

(a) Linear probe, about 550K parameters. Every item is scored on its own, so
    all cross-item context has to have come from the backbone's own attention
    during prefill. Figure 2 draws it as Linear(D, 512), Linear(512, K), which
    at the paper's D = 1024 and K = 50 is 524,800 + 25,650 = 550,450. No
    activation is drawn between the two, but stacking bare linears collapses to
    a single one, so there is a GELU in between here.

(b) Self-attention, L = 2, about 4.2M parameters. Items attend to each other
    before being scored, which is the explicit pairwise comparison of Section
    4.1 and the configuration that wins Table 4. Two pre-norm encoder blocks at
    width 512 with an MLP ratio of 2 come to 2 * (4 * 512^2 + 2 * 512 * 1024) =
    4,194,304, which is where the 4.2M in Table 1 comes from; the projections
    around them are counted separately there.

(c) Slot-query, about 4.0M parameters. K learnable position embeddings
    cross-attend over the items, so the question asked is "what fills rank 1?".
    Scores come out as a dot product between refined slots and projected items.
    Table 1's 4.0M does not fall out of any layer count I could pin down; two
    cross-attention blocks at the same width give 4.19M, so that is the reading
    used here.

Params are plain nested dicts of arrays. Every ``apply`` handles one slate,
shape ``(N, D)`` in and ``(N, K)`` out; ``jax.vmap`` it for a batch.

Heads (b) and (c) are equivariant to a relabelling of the items: permute the
rows of h and the rows of M permute the same way. There is no positional
encoding over items, since a slate is a set and the position information lives
in the K output columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["HeadConfig", "init_head", "apply_head", "count_params", "HEAD_KINDS"]

HEAD_KINDS = ("linear_probe", "self_attention", "slot_query")


@dataclass(frozen=True)
class HeadConfig:
    """Shape of a scoring head. Defaults are the paper's, Figure 2 and Table 1."""

    kind: str = "self_attention"
    d_model: int = 1024  # backbone hidden size D
    n_positions: int = 50  # K
    width: int = 512
    n_layers: int = 2  # L
    n_heads: int = 8
    mlp_ratio: int = 2

    def __post_init__(self):
        if self.kind not in HEAD_KINDS:
            raise ValueError(f"unknown head {self.kind!r}, expected one of {HEAD_KINDS}")
        if self.width % self.n_heads:
            raise ValueError("width must divide evenly among the attention heads")


def _init_linear(key, fan_in, fan_out):
    scale = (1.0 / fan_in) ** 0.5
    w = jax.random.uniform(key, (fan_in, fan_out), minval=-scale, maxval=scale)
    return {"w": w, "b": jnp.zeros((fan_out,))}


def _linear(p, x):
    return x @ p["w"] + p["b"]


def _layer_norm(x, p, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return p["scale"] * (x - mu) * jax.lax.rsqrt(var + eps) + p["shift"]


def _init_norm(width):
    return {"scale": jnp.ones((width,)), "shift": jnp.zeros((width,))}


def _init_attention(key, width, n_heads):
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {
        "q": _init_linear(kq, width, width),
        "k": _init_linear(kk, width, width),
        "v": _init_linear(kv, width, width),
        "out": _init_linear(ko, width, width),
    }


def _attention(p, q_in, kv_in, n_heads):
    """Multi-head attention, no mask. Queries from ``q_in``, keys and values
    from ``kv_in``; passing the same array for both makes it self-attention."""
    nq, width = q_in.shape
    nk = kv_in.shape[0]
    head_dim = width // n_heads
    q = _linear(p["q"], q_in).reshape(nq, n_heads, head_dim).transpose(1, 0, 2)
    k = _linear(p["k"], kv_in).reshape(nk, n_heads, head_dim).transpose(1, 0, 2)
    v = _linear(p["v"], kv_in).reshape(nk, n_heads, head_dim).transpose(1, 0, 2)
    logits = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(logits, axis=-1)
    ctx = jnp.einsum("hqk,hkd->hqd", weights, v).transpose(1, 0, 2).reshape(nq, width)
    return _linear(p["out"], ctx)


def _init_block(key, width, n_heads, mlp_ratio):
    ka, k1, k2 = jax.random.split(key, 3)
    hidden = width * mlp_ratio
    return {
        "norm_attn": _init_norm(width),
        "norm_mem": _init_norm(width),
        "attn": _init_attention(ka, width, n_heads),
        "norm_mlp": _init_norm(width),
        "mlp_in": _init_linear(k1, width, hidden),
        "mlp_out": _init_linear(k2, hidden, width),
    }


def _block(p, x, memory, n_heads):
    """Pre-norm transformer block. ``memory is x`` gives a self-attention layer,
    a different ``memory`` gives a cross-attention layer."""
    h = _layer_norm(x, p["norm_attn"])
    m = h if memory is None else _layer_norm(memory, p["norm_mem"])
    x = x + _attention(p["attn"], h, m, n_heads)
    h = _layer_norm(x, p["norm_mlp"])
    return x + _linear(p["mlp_out"], jax.nn.gelu(_linear(p["mlp_in"], h)))


def init_head(cfg: HeadConfig, key):
    """Fresh parameters for a head."""
    keys = list(jax.random.split(key, cfg.n_layers + 3))
    params = {"proj": _init_linear(keys[0], cfg.d_model, cfg.width)}

    if cfg.kind == "linear_probe":
        params["score"] = _init_linear(keys[1], cfg.width, cfg.n_positions)
        return params

    if cfg.kind == "self_attention":
        params["blocks"] = [
            _init_block(keys[2 + i], cfg.width, cfg.n_heads, cfg.mlp_ratio)
            for i in range(cfg.n_layers)
        ]
        params["norm_out"] = _init_norm(cfg.width)
        params["score"] = _init_linear(keys[1], cfg.width, cfg.n_positions)
        return params

    # slot_query
    params["slots"] = 0.02 * jax.random.normal(keys[1], (cfg.n_positions, cfg.width))
    params["blocks"] = [
        _init_block(keys[2 + i], cfg.width, cfg.n_heads, cfg.mlp_ratio)
        for i in range(cfg.n_layers)
    ]
    params["norm_slots"] = _init_norm(cfg.width)
    params["norm_items"] = _init_norm(cfg.width)
    return params


def apply_head(cfg: HeadConfig, params, hidden):
    """Map per-item hidden states ``(N, D)`` to the score matrix M, ``(N, K)``."""
    hidden = jnp.asarray(hidden)
    if hidden.ndim != 2:
        raise ValueError(f"expected one slate of shape (N, D), got {hidden.shape}")
    if hidden.shape[-1] != cfg.d_model:
        raise ValueError(f"expected D = {cfg.d_model}, got {hidden.shape[-1]}")

    x = _linear(params["proj"], hidden)

    if cfg.kind == "linear_probe":
        # Nothing here mixes across the item axis, on purpose: this head is the
        # ablation that isolates how much comparison the backbone already did.
        return _linear(params["score"], jax.nn.gelu(x))

    if cfg.kind == "self_attention":
        for block in params["blocks"]:
            x = _block(block, x, None, cfg.n_heads)
        return _linear(params["score"], _layer_norm(x, params["norm_out"]))

    # Slot-query: the K position embeddings are the queries and the items are
    # the keys and values, so information flows positions <- items, and M is
    # read off as a scaled dot product between the two.
    slots = params["slots"]
    for block in params["blocks"]:
        slots = _block(block, slots, x, cfg.n_heads)
    slots = _layer_norm(slots, params["norm_slots"])
    items = _layer_norm(x, params["norm_items"])
    return items @ slots.T / jnp.sqrt(cfg.width)


def count_params(params) -> int:
    """Total number of scalars in a parameter tree."""
    return int(sum(x.size for x in jax.tree.leaves(params)))
