"""A small transformer encoder classifier whose forward pass exposes the
residual-stream decomposition of Section 4 of arXiv:2609.05074.

The paper writes the final residual state as

    r^(L) = r^(0) + sum_k (A_k + MLP_k),   A_k = sum_h H_{k,h}

with the class logits read off by a linear map, logit(x) = W^T r^(L). This
module keeps every one of those terms as an array so the decomposition can be
checked numerically instead of assumed.

H_{k,h} is the head's write back into the residual stream, that is the head's
output slice already multiplied by its slice of W_O, not the pre-projection
value vector. Attention has no output bias here, so A_k is exactly the sum of
the per-head writes.

The readout reads one position, `cfg.cls_index`, which is what makes the logit
a linear function of a single residual vector and the decomposition a statement
about one column of the stream. It does have a bias; `readout_bias_diff` is the
constant the residual terms cannot account for.

Pre-LayerNorm blocks leave the residual stream additive, so the decomposition
stays exact with `use_layernorm=True`. What breaks it is a LayerNorm applied
between the final residual state and the readout, since that makes the logit a
nonlinear function of r^(L). That one is `use_final_layernorm` and it is off by
default; turning it on reproduces the approximation the paper makes when it
abstracts normalization away.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

PRECISION = jax.lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16
    seq_len: int = 16
    n_layers: int = 2
    n_heads: int = 3
    d_model: int = 48
    d_head: int = 16
    d_mlp: int = 96
    n_classes: int = 2
    use_layernorm: bool = True
    use_final_layernorm: bool = False
    cls_index: int = 0


class Trace(NamedTuple):
    """Every term of the Section 4 decomposition, for one sequence."""

    embed: jnp.ndarray           # [T, D]        r^(0)
    resid_pre_attn: jnp.ndarray  # [L, T, D]     residual state entering block k
    head_writes: jnp.ndarray     # [L, H, T, D]  H_{k,h}
    attn_out: jnp.ndarray        # [L, T, D]     A_k
    resid_mid: jnp.ndarray       # [L, T, D]     residual state after A_k
    mlp_out: jnp.ndarray         # [L, T, D]     MLP_k
    resid_final: jnp.ndarray     # [T, D]        r^(L)
    logits: jnp.ndarray          # [n_classes]


def init_params(key, cfg: ModelConfig, dtype=jnp.float32):
    keys = jax.random.split(key, 9)
    s = 1.0 / jnp.sqrt(jnp.asarray(cfg.d_model, dtype))

    def normal(k, shape, scale):
        return (scale * jax.random.normal(k, shape, dtype)).astype(dtype)

    return {
        "tok_embed": normal(keys[0], (cfg.vocab_size, cfg.d_model), 0.3),
        "pos_embed": normal(keys[1], (cfg.seq_len, cfg.d_model), 0.3),
        "attn": {
            "wq": normal(keys[2], (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head), s),
            "wk": normal(keys[3], (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head), s),
            "wv": normal(keys[4], (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head), s),
            "wo": normal(keys[5], (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model), s),
        },
        "mlp": {
            "w1": normal(keys[6], (cfg.n_layers, cfg.d_model, cfg.d_mlp), s),
            "b1": jnp.zeros((cfg.n_layers, cfg.d_mlp), dtype),
            "w2": normal(keys[7], (cfg.n_layers, cfg.d_mlp, cfg.d_model), s),
            "b2": jnp.zeros((cfg.n_layers, cfg.d_model), dtype),
        },
        "ln1": {"gain": jnp.ones((cfg.n_layers, cfg.d_model), dtype),
                "bias": jnp.zeros((cfg.n_layers, cfg.d_model), dtype)},
        "ln2": {"gain": jnp.ones((cfg.n_layers, cfg.d_model), dtype),
                "bias": jnp.zeros((cfg.n_layers, cfg.d_model), dtype)},
        "lnf": {"gain": jnp.ones((cfg.d_model,), dtype),
                "bias": jnp.zeros((cfg.d_model,), dtype)},
        "readout": {"w": normal(keys[8], (cfg.n_classes, cfg.d_model), s),
                    "b": jnp.zeros((cfg.n_classes,), dtype)},
    }


def _layer_norm(x, gain, bias, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    centred = x - mu
    var = (centred ** 2).mean(-1, keepdims=True)
    return centred * jax.lax.rsqrt(var + eps) * gain + bias


def _attention(cfg: ModelConfig, p, x, head_mask):
    """Per-head residual-stream writes H_{k,h} for one block.

    Returns [H, T, D]; summing over the head axis gives A_k.
    """
    q = jnp.einsum("td,hde->hte", x, p["wq"], precision=PRECISION)
    k = jnp.einsum("td,hde->hte", x, p["wk"], precision=PRECISION)
    v = jnp.einsum("td,hde->hte", x, p["wv"], precision=PRECISION)
    scale = 1.0 / jnp.sqrt(jnp.asarray(cfg.d_head, x.dtype))
    scores = jnp.einsum("hte,hse->hts", q, k, precision=PRECISION) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    z = jnp.einsum("hts,hse->hte", weights, v, precision=PRECISION)
    writes = jnp.einsum("hte,hed->htd", z, p["wo"], precision=PRECISION)
    return writes * head_mask[:, None, None]


def _mlp(p, x):
    h = jax.nn.gelu(jnp.einsum("td,dm->tm", x, p["w1"], precision=PRECISION) + p["b1"])
    return jnp.einsum("tm,md->td", h, p["w2"], precision=PRECISION) + p["b2"]


@functools.partial(jax.jit, static_argnums=0)
def forward(cfg: ModelConfig, params, tokens, head_mask=None):
    """Run the encoder on one token sequence and keep every residual term.

    `head_mask` is [n_layers, n_heads]; a zero entry deletes that head's write
    to the residual stream, which is the ablation used in Section 6.4.
    """
    dtype = params["tok_embed"].dtype
    if head_mask is None:
        head_mask = jnp.ones((cfg.n_layers, cfg.n_heads), dtype)
    head_mask = head_mask.astype(dtype)

    resid = params["tok_embed"][tokens] + params["pos_embed"][: tokens.shape[0]]
    embed = resid

    pre_attn, writes_all, attn_all, mid_all, mlp_all = [], [], [], [], []
    for k in range(cfg.n_layers):
        attn_p = jax.tree.map(lambda a: a[k], params["attn"])
        mlp_p = jax.tree.map(lambda a: a[k], params["mlp"])
        pre_attn.append(resid)
        x = _layer_norm(resid, params["ln1"]["gain"][k], params["ln1"]["bias"][k]) if cfg.use_layernorm else resid
        writes = _attention(cfg, attn_p, x, head_mask[k])
        attn_out = writes.sum(0)
        resid = resid + attn_out
        mid_all.append(resid)
        y = _layer_norm(resid, params["ln2"]["gain"][k], params["ln2"]["bias"][k]) if cfg.use_layernorm else resid
        mlp_out = _mlp(mlp_p, y)
        resid = resid + mlp_out
        writes_all.append(writes)
        attn_all.append(attn_out)
        mlp_all.append(mlp_out)

    readout_in = resid
    if cfg.use_final_layernorm:
        readout_in = _layer_norm(readout_in, params["lnf"]["gain"], params["lnf"]["bias"])
    logits = params["readout"]["w"] @ readout_in[cfg.cls_index] + params["readout"]["b"]

    return Trace(
        embed=embed,
        resid_pre_attn=jnp.stack(pre_attn),
        head_writes=jnp.stack(writes_all),
        attn_out=jnp.stack(attn_all),
        resid_mid=jnp.stack(mid_all),
        mlp_out=jnp.stack(mlp_all),
        resid_final=resid,
        logits=logits,
    )


def logits_only(cfg: ModelConfig, params, tokens, head_mask=None):
    return forward(cfg, params, tokens, head_mask).logits


batched_logits = jax.vmap(logits_only, in_axes=(None, None, 0, None))


def logit_direction(params, positive: int = 1, negative: int = 0) -> jnp.ndarray:
    """w = W[class_positive] - W[class_negative], the direction of Section 4."""
    return params["readout"]["w"][positive] - params["readout"]["w"][negative]


def readout_bias_diff(params, positive: int = 1, negative: int = 0):
    """The part of the logit difference the residual decomposition cannot see."""
    return params["readout"]["b"][positive] - params["readout"]["b"][negative]
