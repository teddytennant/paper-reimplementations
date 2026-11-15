"""Small sequence denoiser in pure JAX.

Equinox and Flax are not required. One pre-norm transformer block, a sinusoidal
time embedding added to token and position embeddings, and a linear head.
The mask logit is set to -inf (SUBS) so the softmax has zero mass on the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from refinemix.loss import subs_logits


@dataclass(frozen=True)
class DenoiserConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_heads: int
    mask_id: int


def init_denoiser(
    key,
    vocab_size: int,
    seq_len: int,
    d_model: int = 32,
    n_heads: int = 2,
    mask_id: int | None = None,
    mlp_ratio: int = 2,
):
    if mask_id is None:
        mask_id = vocab_size - 1
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for the sinusoidal time embedding")
    k = jax.random.split(key, 8)
    scale = 0.02

    def n(i, shape):
        return jax.random.normal(k[i], shape) * scale

    params = {
        "tok": n(0, (vocab_size, d_model)),
        "pos": n(1, (seq_len, d_model)),
        "t1": n(2, (d_model, d_model)),
        "t2": n(3, (d_model, d_model)),
        "qkv": n(4, (d_model, 3 * d_model)),
        "attn_out": n(5, (d_model, d_model)),
        "ff1": n(6, (d_model, mlp_ratio * d_model)),
        "ff2": n(7, (mlp_ratio * d_model, d_model)),
        "out": n(0, (d_model, vocab_size)),
        "ln1": jnp.ones((d_model,)),
        "ln2": jnp.ones((d_model,)),
    }
    cfg = DenoiserConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        mask_id=int(mask_id),
    )
    return params, cfg


def _layernorm(x, scale, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale


def _time_embed(t, d_model: int):
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:
        t = t[None]
    half = d_model // 2
    freqs = jnp.exp(-jnp.arange(half, dtype=jnp.float32) * (jnp.log(10000.0) / half))
    ang = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def apply_denoiser(params, tokens, t, *, n_heads: int, mask_id: int):
    """Logits (B, L, V) with the mask logit set to -inf.

    `tokens` is (B, L) integer. `t` is a scalar or (B,).
    """
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    b, length = tokens.shape
    d_model = params["tok"].shape[1]
    x = params["tok"][tokens] + params["pos"][:length][None, :, :]
    te = _time_embed(t, d_model)
    if te.shape[0] == 1 and b > 1:
        te = jnp.broadcast_to(te, (b, d_model))
    te = jax.nn.gelu(te @ params["t1"]) @ params["t2"]
    x = x + te[:, None, :]

    h = _layernorm(x, params["ln1"])
    qkv = h @ params["qkv"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    dh = d_model // n_heads

    def split_heads(z):
        return z.reshape(b, length, n_heads, dh).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    att = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(dh, dtype=q.dtype))
    att = jax.nn.softmax(att, axis=-1)
    mixed = jnp.matmul(att, v).transpose(0, 2, 1, 3).reshape(b, length, d_model)
    x = x + mixed @ params["attn_out"]
    h = _layernorm(x, params["ln2"])
    x = x + jax.nn.gelu(h @ params["ff1"]) @ params["ff2"]
    logits = x @ params["out"]
    return subs_logits(logits, mask_id)


def make_apply_fn(n_heads: int, mask_id: int):
    def apply_fn(params, tokens, t):
        return apply_denoiser(params, tokens, t, n_heads=n_heads, mask_id=mask_id)

    return apply_fn
