"""AdaLN (Eq. 9), multi-head attention (Eq. 10), block-causal mask, FFN."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from jax_flowatc.config import FlowATCConfig

Array = jax.Array


def adaln(h: Array, gamma: Array, beta: Array, eps: float = 1e-6) -> Array:
    """AdaLN(h, t) = γ(t) * (h - μ(h)) / σ(h) + β(t)  (Eq. 9).

    μ, σ are over the feature axis, per token. `gamma` and `beta` broadcast
    over the token axis.
    """
    mu = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean((h - mu) ** 2, axis=-1, keepdims=True)
    sigma = jnp.sqrt(var + eps)
    return gamma * (h - mu) / sigma + beta


def adaln_from_stats(h: Array, mu: Array, sigma: Array, gamma: Array, beta: Array) -> Array:
    """Eq. 9 with caller-supplied μ, σ (no eps)."""
    return gamma * (h - mu) / sigma + beta


def block_causal_mask(seq_len: int, t_obs: int) -> Array:
    """Additive mask A ∈ {0, -∞}^{T×T}.

    A_ij = -∞ iff i < T_obs AND j ≥ T_obs, else 0.
    Observed tokens cannot attend to noisy future tokens; future tokens attend
    to the observed prefix and to one another.
    """
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]
    forbidden = (i < t_obs) & (j >= t_obs)
    return jnp.where(forbidden, -jnp.inf, 0.0)


def sinusoidal_embedding(times: Array, dim: int, max_period: float = 10000.0) -> Array:
    """Transformer sinusoidal features of a scalar time per token (or a scalar t)."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = times[..., None] * freqs
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
    return emb


def dropout(x: Array, rate: float, key: Optional[Array]) -> Array:
    if rate == 0.0 or key is None:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def attention(
    z: Array,
    w_q: Array,
    w_k: Array,
    w_v: Array,
    w_o: Array,
    mask: Array,
    n_heads: int,
) -> Array:
    """Multi-head self-attention (Eq. 10). z: (T, d) -> (T, d)."""
    t_len, d_model = z.shape
    head_dim = d_model // n_heads
    scale = jnp.sqrt(jnp.asarray(head_dim, dtype=z.dtype))
    q = (z @ w_q).reshape(t_len, n_heads, head_dim).transpose(1, 0, 2)
    k = (z @ w_k).reshape(t_len, n_heads, head_dim).transpose(1, 0, 2)
    v = (z @ w_v).reshape(t_len, n_heads, head_dim).transpose(1, 0, 2)
    logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / scale
    logits = logits + mask[None, :, :]
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.matmul(weights, v)
    out = out.transpose(1, 0, 2).reshape(t_len, d_model)
    return out @ w_o


def attention_weights(
    z: Array,
    w_q: Array,
    w_k: Array,
    mask: Array,
    n_heads: int,
) -> Array:
    """Softmax attention weights, shape (H, T, T)."""
    t_len, d_model = z.shape
    head_dim = d_model // n_heads
    scale = jnp.sqrt(jnp.asarray(head_dim, dtype=z.dtype))
    q = (z @ w_q).reshape(t_len, n_heads, head_dim).transpose(1, 0, 2)
    k = (z @ w_k).reshape(t_len, n_heads, head_dim).transpose(1, 0, 2)
    logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / scale
    logits = logits + mask[None, :, :]
    return jax.nn.softmax(logits, axis=-1)


def ffn(h: Array, w1: Array, b1: Array, w2: Array, b2: Array) -> Array:
    """Position-wise FFN with expansion factor 4 and GELU."""
    return jax.nn.gelu(h @ w1 + b1) @ w2 + b2


def adaln_modulation(t_emb: Array, w1: Array, b1: Array, w2: Array, b2: Array):
    """Small MLP: t_emb -> (γ, β) each in R^d."""
    hidden = jax.nn.silu(t_emb @ w1 + b1)
    out = hidden @ w2 + b2
    gamma, beta = jnp.split(out, 2, axis=-1)
    return gamma, beta


def transformer_block(
    z: Array,
    t_emb: Array,
    mask: Array,
    block,
    cfg: FlowATCConfig,
    dropout_keys: Optional[tuple] = None,
) -> Array:
    """One DiT block: AdaLN-attn residual, then AdaLN-FFN residual."""
    k_attn = dropout_keys[0] if dropout_keys is not None else None
    k_ffn = dropout_keys[1] if dropout_keys is not None else None
    gamma_a, beta_a = adaln_modulation(
        t_emb, block.adaln_attn.w1, block.adaln_attn.b1, block.adaln_attn.w2, block.adaln_attn.b2
    )
    h = adaln(z, gamma_a, beta_a, cfg.ln_eps)
    h = attention(h, block.attn.w_q, block.attn.w_k, block.attn.w_v, block.attn.w_o, mask, cfg.n_heads)
    z = z + dropout(h, cfg.dropout, k_attn)
    gamma_f, beta_f = adaln_modulation(
        t_emb, block.adaln_ffn.w1, block.adaln_ffn.b1, block.adaln_ffn.w2, block.adaln_ffn.b2
    )
    h = adaln(z, gamma_f, beta_f, cfg.ln_eps)
    h = ffn(h, block.ffn.w1, block.ffn.b1, block.ffn.w2, block.ffn.b2)
    z = z + dropout(h, cfg.dropout, k_ffn)
    return z
