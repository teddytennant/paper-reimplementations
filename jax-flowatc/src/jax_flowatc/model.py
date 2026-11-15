"""DiT backbone: Eq. 11 concat, time embeddings, AdaLN blocks, future readout."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_flowatc.config import FlowATCConfig
from jax_flowatc.layers import (
    adaln,
    adaln_modulation,
    attention_weights,
    block_causal_mask,
    sinusoidal_embedding,
    transformer_block,
)

Array = jax.Array


class AdaLNParams(NamedTuple):
    w1: Array
    b1: Array
    w2: Array
    b2: Array


class AttnParams(NamedTuple):
    w_q: Array
    w_k: Array
    w_v: Array
    w_o: Array


class FFNParams(NamedTuple):
    w1: Array
    b1: Array
    w2: Array
    b2: Array


class BlockParams(NamedTuple):
    attn: AttnParams
    ffn: FFNParams
    adaln_attn: AdaLNParams
    adaln_ffn: AdaLNParams


class FlowATCParams(NamedTuple):
    w_in: Array
    w_out: Array
    dt_w: Array
    dt_b: Array
    time_w1: Array
    time_b1: Array
    time_w2: Array
    time_b2: Array
    t_proj_w: Array
    t_proj_b: Array
    blocks: Tuple[BlockParams, ...]
    final_adaln: AdaLNParams


def _linear(key: Array, in_dim: int, out_dim: int, scale: Optional[float] = None) -> Array:
    if scale is None:
        scale = in_dim ** -0.5
    return jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * scale


def _adaln_params(key: Array, d: int) -> AdaLNParams:
    k1, k2 = jax.random.split(key)
    return AdaLNParams(
        w1=_linear(k1, d, d),
        b1=jnp.zeros((d,), dtype=jnp.float32),
        w2=_linear(k2, d, 2 * d),
        b2=jnp.zeros((2 * d,), dtype=jnp.float32),
    )


def _block_params(key: Array, cfg: FlowATCConfig) -> BlockParams:
    d = cfg.d_model
    keys = jax.random.split(key, 8)
    attn = AttnParams(
        w_q=_linear(keys[0], d, d),
        w_k=_linear(keys[1], d, d),
        w_v=_linear(keys[2], d, d),
        w_o=_linear(keys[3], d, d),
    )
    ffn_h = cfg.ffn_mult * d
    ffn = FFNParams(
        w1=_linear(keys[4], d, ffn_h),
        b1=jnp.zeros((ffn_h,), dtype=jnp.float32),
        w2=_linear(keys[5], ffn_h, d),
        b2=jnp.zeros((d,), dtype=jnp.float32),
    )
    return BlockParams(
        attn=attn,
        ffn=ffn,
        adaln_attn=_adaln_params(keys[6], d),
        adaln_ffn=_adaln_params(keys[7], d),
    )


def init_params(key: Array, cfg: FlowATCConfig) -> FlowATCParams:
    d = cfg.d_model
    keys = jax.random.split(key, 6 + cfg.n_layers)
    w_in = _linear(keys[0], cfg.state_dim, d)
    w_out = _linear(keys[1], d, cfg.state_dim)
    dt_w = _linear(keys[2], 1, d)
    dt_b = jnp.zeros((d,), dtype=jnp.float32)
    time_w1 = _linear(keys[3], d, d)
    time_b1 = jnp.zeros((d,), dtype=jnp.float32)
    time_w2 = _linear(keys[4], d, d)
    time_b2 = jnp.zeros((d,), dtype=jnp.float32)
    t_proj_w = _linear(keys[5], d, d)
    t_proj_b = jnp.zeros((d,), dtype=jnp.float32)
    blocks = tuple(_block_params(keys[6 + i], cfg) for i in range(cfg.n_layers))
    final_key = jax.random.fold_in(key, 17)
    return FlowATCParams(
        w_in=w_in,
        w_out=w_out,
        dt_w=dt_w,
        dt_b=dt_b,
        time_w1=time_w1,
        time_b1=time_b1,
        time_w2=time_w2,
        time_b2=time_b2,
        t_proj_w=t_proj_w,
        t_proj_b=t_proj_b,
        blocks=blocks,
        final_adaln=_adaln_params(final_key, d),
    )


def eq11_concat(w_in: Array, x_obs: Array, x_tilde_fut: Array) -> Array:
    """Z = [W_in X_obs || W_in X_tilde_fut] ∈ R^{T × d}  (Eq. 11)."""
    return jnp.concatenate([x_obs @ w_in, x_tilde_fut @ w_in], axis=-2)


def _add_time_embeddings(
    z: Array,
    dt: Array,
    t_rel: Array,
    params: FlowATCParams,
    cfg: FlowATCConfig,
) -> Array:
    dt_emb = dt[..., None] @ params.dt_w + params.dt_b
    sin = sinusoidal_embedding(t_rel, cfg.d_model, cfg.time_max_period)
    t_tok = jax.nn.silu(sin @ params.time_w1 + params.time_b1)
    t_tok = t_tok @ params.time_w2 + params.time_b2
    return z + dt_emb + t_tok


def flow_time_embedding(t: Array, params: FlowATCParams, cfg: FlowATCConfig) -> Array:
    """Sinusoidal embedding of CFM time t ∈ [0, 1], projected to d_model."""
    sin = sinusoidal_embedding(jnp.asarray(t, dtype=jnp.float32), cfg.d_model, cfg.time_max_period)
    return jax.nn.silu(sin @ params.t_proj_w + params.t_proj_b)


def embed_tokens(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_tilde_fut: Array,
    dt: Array,
    t_rel: Array,
) -> Array:
    """Eq. 11 projection plus Δt and elapsed-time embeddings. Returns (T, d)."""
    z = eq11_concat(params.w_in, x_obs, x_tilde_fut)
    return _add_time_embeddings(z, dt, t_rel, params, cfg)


def hidden_states(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_tilde_fut: Array,
    t: Array,
    dt: Array,
    t_rel: Array,
    dropout_key: Optional[Array] = None,
) -> Array:
    """Full DiT stack hidden states, shape (T, d)."""
    z = embed_tokens(params, cfg, x_obs, x_tilde_fut, dt, t_rel)
    t_emb = flow_time_embedding(t, params, cfg)
    mask = block_causal_mask(cfg.seq_len, cfg.t_obs)
    keys = None
    if dropout_key is not None and cfg.dropout > 0.0:
        keys = jax.random.split(dropout_key, cfg.n_layers * 2)
    for i, block in enumerate(params.blocks):
        dkeys = None if keys is None else (keys[2 * i], keys[2 * i + 1])
        z = transformer_block(z, t_emb, mask, block, cfg, dkeys)
    return z


def velocity_field(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_tilde_fut: Array,
    t: Array,
    dt: Array,
    t_rel: Array,
    dropout_key: Optional[Array] = None,
) -> Array:
    """Predicted velocity on the T_fut future tokens, shape (T_fut, state_dim)."""
    z = hidden_states(params, cfg, x_obs, x_tilde_fut, t, dt, t_rel, dropout_key)
    t_emb = flow_time_embedding(t, params, cfg)
    gamma, beta = adaln_modulation(
        t_emb,
        params.final_adaln.w1,
        params.final_adaln.b1,
        params.final_adaln.w2,
        params.final_adaln.b2,
    )
    z = adaln(z, gamma, beta, cfg.ln_eps)
    fut = z[cfg.t_obs :]
    return fut @ params.w_out


def block_attention_weights(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_tilde_fut: Array,
    t: Array,
    dt: Array,
    t_rel: Array,
    layer: int = 0,
) -> Array:
    """Attention weights of one block, shape (H, T, T)."""
    z = embed_tokens(params, cfg, x_obs, x_tilde_fut, dt, t_rel)
    t_emb = flow_time_embedding(t, params, cfg)
    mask = block_causal_mask(cfg.seq_len, cfg.t_obs)
    for i, block in enumerate(params.blocks):
        if i == layer:
            gamma, beta = adaln_modulation(
                t_emb, block.adaln_attn.w1, block.adaln_attn.b1, block.adaln_attn.w2, block.adaln_attn.b2
            )
            h = adaln(z, gamma, beta, cfg.ln_eps)
            return attention_weights(h, block.attn.w_q, block.attn.w_k, mask, cfg.n_heads)
        z = transformer_block(z, t_emb, mask, block, cfg, None)
    raise ValueError(f"layer {layer} out of range")


def apply_one_block(
    params: FlowATCParams,
    cfg: FlowATCConfig,
    x_obs: Array,
    x_tilde_fut: Array,
    t: Array,
    dt: Array,
    t_rel: Array,
    layer: int = 0,
) -> Array:
    """Hidden states after a single Transformer block, shape (T, d)."""
    z = embed_tokens(params, cfg, x_obs, x_tilde_fut, dt, t_rel)
    t_emb = flow_time_embedding(t, params, cfg)
    mask = block_causal_mask(cfg.seq_len, cfg.t_obs)
    return transformer_block(z, t_emb, mask, params.blocks[layer], cfg, None)
