"""Tiny synthetic LM: train a linear next-token model with TF-IDF CE."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from tfidf_ce.buffer import IdfBuffer
from tfidf_ce.loss import tfidf_cross_entropy

VOCAB = 32
LENGTH = 16
BATCH = 8
K = 4
STEPS = 200
LR = 0.2
PAD_ID = 0
STOP = 1  # appears in every sequence
CONTENT_A = 20
CONTENT_B = 21
LABEL_A = 22
LABEL_B = 23
D_MODEL = 16


def make_batch(key: jax.Array) -> jax.Array:
    """Stopwords plus a content token that determines the last token."""
    k_fill, k_lab, k_pad = jax.random.split(key, 3)
    tokens = jax.random.randint(k_fill, (BATCH, LENGTH), minval=2, maxval=12)
    tokens = tokens.at[:, :4].set(STOP)
    labels = jax.random.bernoulli(k_lab, 0.5, (BATCH,)).astype(jnp.int32)
    tokens = tokens.at[:, 8].set(jnp.where(labels == 0, CONTENT_A, CONTENT_B))
    tokens = tokens.at[:, -1].set(jnp.where(labels == 0, LABEL_A, LABEL_B))
    # Pad the last two positions of half the batch so M < B*L.
    pad_row = jax.random.bernoulli(k_pad, 0.5, (BATCH,)).astype(bool)
    tokens = jnp.where(pad_row[:, None] & (jnp.arange(LENGTH) >= LENGTH - 2), PAD_ID, tokens)
    return tokens.astype(jnp.int32)


def init_params(key: jax.Array) -> dict:
    k_w, k_b = jax.random.split(key)
    return {
        "embed": jax.random.normal(k_w, (VOCAB, D_MODEL)) * 0.1,
        "w": jax.random.normal(k_b, (D_MODEL, VOCAB)) * 0.1,
        "b": jnp.zeros((VOCAB,)),
    }


def logits_fn(params: dict, tokens: jax.Array) -> jax.Array:
    """Next-token logits of shape [B, L-1, V] from tokens [B, L]."""
    hidden = params["embed"][tokens[:, :-1]]
    return hidden @ params["w"] + params["b"]


def batch_loss(params: dict, tokens: jax.Array, weights: jax.Array) -> jax.Array:
    logits = logits_fn(params, tokens)
    return tfidf_cross_entropy(
        logits, tokens[:, 1:], weights[:, 1:], pad_id=PAD_ID
    )


def sgd_step(params: dict, tokens: jax.Array, weights: jax.Array, lr: float) -> dict:
    grads = jax.grad(batch_loss)(params, tokens, weights)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def main() -> int:
    key = jax.random.PRNGKey(0)
    key, k_init = jax.random.split(key)
    params = init_params(k_init)
    buf = IdfBuffer(k=K, vocab_size=VOCAB, pad_id=PAD_ID)

    key, k_batch = jax.random.split(key)
    first = make_batch(k_batch)
    w0 = buf.update(first)
    loss_before = float(batch_loss(params, first, w0))

    for step in range(STEPS):
        key, k_batch = jax.random.split(key)
        tokens = make_batch(k_batch)
        weights = buf.update(tokens)
        params = sgd_step(params, tokens, weights, LR)

    key, k_batch = jax.random.split(key)
    last = make_batch(k_batch)
    w1 = buf.update(last)
    loss_after = float(batch_loss(params, last, w1))

    print(f"loss_before={loss_before:.6f}")
    print(f"loss_after={loss_after:.6f}")
    if not (loss_after < loss_before):
        print("loss did not drop", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
