"""The masked Transformer layer of Section II-B.

The paper writes embeddings as E in R^(l x d): l is the model width and d the
number of tokens, so tokens are columns and attention is

    Attention(Q, K, V) = V^T softmax( K^T Q / sqrt(l) + M )                (8)

with softmax down each column. This module uses the ordinary row-token layout,
x in R^(d x l), which puts the same computation in the familiar form

    softmax( Q K^T / sqrt(d_head) + M ) V

with softmax along the last axis. The mask index convention is preserved: in
either layout, M[i, j] = -inf stops token j from reading token i, and the masks
built by LinearBlockCode.attention_mask are symmetric, so the distinction is not
load-bearing here anyway.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class MultiHeadAttention(eqx.Module):
    """Standard multi-head self-attention with an additive attention mask."""

    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    out: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, width: int, n_heads: int, *, key: jax.Array):
        if width % n_heads:
            raise ValueError(f"width {width} not divisible by n_heads {n_heads}")
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q = eqx.nn.Linear(width, width, use_bias=False, key=kq)
        self.k = eqx.nn.Linear(width, width, use_bias=False, key=kk)
        self.v = eqx.nn.Linear(width, width, use_bias=False, key=kv)
        self.out = eqx.nn.Linear(width, width, use_bias=False, key=ko)
        self.n_heads = n_heads
        self.d_head = width // n_heads

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """x is (tokens, width); mask is (tokens, tokens) of 0 and -inf."""
        tokens = x.shape[0]

        def split(proj, inp):
            h = jax.vmap(proj)(inp)
            return h.reshape(tokens, self.n_heads, self.d_head).transpose(1, 0, 2)

        q, k, v = split(self.q, x), split(self.k, x), split(self.v, x)
        logits = q @ k.transpose(0, 2, 1) / jnp.sqrt(self.d_head)
        logits = logits + mask[None, :, :]
        # A row of the mask can in principle be all -inf. Not possible for the
        # masks here, since A always has a nonzero diagonal, but softmax of an
        # all -inf row is NaN and that failure is silent, so guard it.
        weights = jax.nn.softmax(logits, axis=-1, where=jnp.isfinite(logits))
        weights = jnp.where(jnp.isfinite(logits), weights, 0.0)
        merged = (weights @ v).transpose(1, 0, 2).reshape(tokens, -1)
        return jax.vmap(self.out)(merged)


class TransformerLayer(eqx.Module):
    """Fig. 2: masked self-attention, residual + LayerNorm, FFN, residual + LayerNorm."""

    attn: MultiHeadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(self, width: int, n_heads: int, mlp_mult: int, *, key: jax.Array):
        ka, k1, k2 = jax.random.split(key, 3)
        self.attn = MultiHeadAttention(width, n_heads, key=ka)
        self.norm1 = eqx.nn.LayerNorm(width)
        self.norm2 = eqx.nn.LayerNorm(width)
        self.ff1 = eqx.nn.Linear(width, width * mlp_mult, key=k1)
        self.ff2 = eqx.nn.Linear(width * mlp_mult, width, key=k2)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        x = jax.vmap(self.norm1)(x + self.attn(x, mask))
        h = jax.vmap(self.ff2)(jax.nn.gelu(jax.vmap(self.ff1)(x)))
        return jax.vmap(self.norm2)(x + h)


class TransformerStack(eqx.Module):
    """Trans^(e) / Trans^(d): a few layers plus a linear head down to one scalar per token.

    The head is what turns the l-dimensional token representations into the
    o in R^(1 x d) that equations (15), (25) and (30) all produce.
    """

    layers: tuple[TransformerLayer, ...]
    head: eqx.nn.Linear

    def __init__(
        self, width: int, n_layers: int, n_heads: int, mlp_mult: int, *, key: jax.Array
    ):
        keys = jax.random.split(key, n_layers + 1)
        self.layers = tuple(
            TransformerLayer(width, n_heads, mlp_mult, key=k) for k in keys[:-1]
        )
        self.head = eqx.nn.Linear(width, 1, key=keys[-1])

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x, mask)
        return jax.vmap(self.head)(x)[:, 0]
