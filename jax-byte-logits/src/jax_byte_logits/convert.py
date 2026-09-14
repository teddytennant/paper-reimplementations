"""Appendix F.1: token probabilities to byte distributions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.nn import softmax

EOT = 256


def token_byte_tables(token_strings, use_eot=False):
    """Build padded byte table and lengths from UTF-8 token strings.

    token_to_bytes has shape (vocab_size, max_len) with -1 padding.
    lengths has shape (vocab_size,).
    When use_eot is True each token is the UTF-8 bytes plus EOT (index 256).
    """
    rows = []
    for tok in token_strings:
        if isinstance(tok, str):
            seq = list(tok.encode("utf-8"))
        elif isinstance(tok, (bytes, bytearray)):
            seq = list(tok)
        else:
            seq = [int(x) for x in tok]
        if use_eot:
            seq = seq + [EOT]
        rows.append(seq)
    max_len = max((len(r) for r in rows), default=0)
    table = np.full((len(rows), max_len), -1, dtype=np.int32)
    lengths = np.zeros((len(rows),), dtype=np.int32)
    for i, seq in enumerate(rows):
        if seq:
            table[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return jnp.asarray(table), jnp.asarray(lengths)


def _as_gold(gold):
    if isinstance(gold, (bytes, bytearray)):
        return jnp.asarray(list(gold), dtype=jnp.int32)
    if isinstance(gold, str):
        return jnp.asarray(list(gold.encode("utf-8")), dtype=jnp.int32)
    return jnp.asarray(gold, dtype=jnp.int32)


def byte_probabilities(probs, token_to_bytes, lengths, gold, use_eot=False):
    """Byte distributions P(b | gold[:k], context) for k = 0..len(gold)-1.

    Faithful JAX form of Appendix F.1.
    probs (vocab_size):                    token probabilities.
    token_to_bytes (vocab_size, max_len): per-token bytes, -1 padded.
    lengths (vocab_size):                  true byte length.
    gold:                                  byte sequence to condition on.
    use_eot:                               if True, alphabet is {0..255} U {EOT} (size 257) else 256.

    Returns (len(gold), n_bytes) stacked distributions.
    """
    probs = jnp.asarray(probs)
    token_to_bytes = jnp.asarray(token_to_bytes)
    lengths = jnp.asarray(lengths)
    gold = _as_gold(gold)
    n_bytes = 257 if use_eot else 256
    live = jnp.ones((probs.shape[0],), dtype=bool)
    dist = []
    gold_len = int(gold.shape[0])
    for k in range(gold_len):
        active = live & (lengths > k)
        bytes_at_k = token_to_bytes[:, k]
        safe_bytes = jnp.where(active, bytes_at_k, 0)
        safe_bytes = jnp.clip(safe_bytes, 0, n_bytes - 1).astype(jnp.int32)
        masked_probs = jnp.where(active, probs, jnp.zeros_like(probs))
        accum = jnp.bincount(safe_bytes, weights=masked_probs, length=n_bytes)
        total = accum.sum()
        dist.append(jnp.where(total > 0, accum / total, accum))
        live = live & (bytes_at_k == gold[k]) & (lengths > k + 1)
    if not dist:
        return jnp.zeros((0, n_bytes))
    return jnp.stack(dist, axis=0)


def from_logits(logits, token_to_bytes, lengths, gold, use_eot=False):
    """Softmax over token logits, then Appendix F.1 conversion."""
    probs = softmax(jnp.asarray(logits), axis=-1)
    return byte_probabilities(
        probs, token_to_bytes, lengths, gold, use_eot=use_eot
    )
