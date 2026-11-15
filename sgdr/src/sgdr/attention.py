"""Block-sparse attention output.

Router scores are unscaled. This module applies 1/sqrt(d) to the token
logits, matching standard attention rather than eq 12.

A query token attends to a key token only if the block mask allows that
key block and the key position is not in the future. With a p=1 mask this
matches dense causal attention. The paper's eq 1 is block-causal and does
not mask future tokens inside the diagonal block; token-level causality is
used here so the p=1 check is against ordinary dense causal attention.
"""

import jax.numpy as jnp


def token_mask_from_blocks(block_mask, seq_len, block_size, token_causal=True):
    """Expand a block mask to tokens. Positions past full blocks are off."""
    block_mask = jnp.asarray(block_mask)
    bsz = int(block_size)
    n_q = block_mask.shape[0]
    n_k = block_mask.shape[1]
    query_block = jnp.arange(seq_len) // bsz
    key_block = jnp.arange(seq_len) // bsz
    valid_q = query_block < n_q
    valid_k = key_block < n_k
    allowed = block_mask[jnp.minimum(query_block, n_q - 1)[:, None], jnp.minimum(key_block, n_k - 1)[None, :]]
    allowed = allowed & valid_q[:, None] & valid_k[None, :]
    if token_causal:
        pos = jnp.arange(seq_len)
        allowed = allowed & (pos[None, :] <= pos[:, None])
    return allowed


def _attend(q, k, v, allowed):
    q = jnp.asarray(q, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    logits = (q @ k.T) * scale
    logits = jnp.where(allowed, logits, -1e30)
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    weights = jnp.where(allowed, weights, 0.0)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-30)
    return weights @ v


def dense_causal_attention(q, k, v):
    """Standard causal attention with 1/sqrt(d). q, k, v are (L, d)."""
    length = q.shape[0]
    pos = jnp.arange(length)
    allowed = pos[None, :] <= pos[:, None]
    return _attend(q, k, v, allowed)


def sparse_block_attention(q, k, v, block_mask, block_size, token_causal=True):
    """Attention on selected blocks only. Applies 1/sqrt(d).

    q, k, v: (L, d). Pass post-RoPE q and k if RoPE is part of the model.
    This function does not apply RoPE itself.
    """
    allowed = token_mask_from_blocks(block_mask, q.shape[0], block_size, token_causal=token_causal)
    if k.shape[0] != q.shape[0]:
        raise ValueError("q and k sequence lengths must match")
    return _attend(q, k, v, allowed)
