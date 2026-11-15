"""A synthetic binary task with a known mechanism.

The paper's classifier and dataset are not public, so the score needs some
model where the right answer is knowable. Here the label is 1 when the token
immediately after the marker is the trigger token, and 0 otherwise. Negatives
carry a trigger elsewhere about 43% of the time, so answering from the
trigger's presence alone tops out near 78%; the last 22 points need the model
to check which token sits next to the marker.

That takes two heads in series, one below to bring the marker and its
neighbour together and one above to carry the verdict to the readout position.
Only the second writes toward the logit, which turns out to decide what the
influence score can and cannot see.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

PAD = 0
CLS = 1
MARKER = 2
TRIGGER = 3
FIRST_FILLER = 4


def make_example(key, seq_len: int = 16, vocab_size: int = 16, distractor_prob: float = 0.5):
    k_fill, k_pos, k_label, k_dist, k_use = jax.random.split(key, 5)
    tokens = jax.random.randint(k_fill, (seq_len,), FIRST_FILLER, vocab_size)
    marker_pos = jax.random.randint(k_pos, (), 1, seq_len - 1)
    label = jax.random.bernoulli(k_label, 0.5).astype(jnp.int32)

    # A trigger somewhere other than right after the marker, so the trigger's
    # presence is a weak cue rather than the answer.
    dist_pos = jax.random.randint(k_dist, (), 1, seq_len)
    use_dist = jax.random.bernoulli(k_use, distractor_prob)
    place_dist = use_dist & (dist_pos != marker_pos) & (dist_pos != marker_pos + 1)
    tokens = jnp.where(place_dist, tokens.at[dist_pos].set(TRIGGER), tokens)

    tokens = tokens.at[marker_pos].set(MARKER)
    tokens = jnp.where(label == 1, tokens.at[marker_pos + 1].set(TRIGGER), tokens)
    tokens = tokens.at[0].set(CLS)
    return tokens, label


def make_batch(key, batch_size: int, seq_len: int = 16, vocab_size: int = 16,
               distractor_prob: float = 0.5):
    keys = jax.random.split(key, batch_size)
    return jax.vmap(lambda k: make_example(k, seq_len, vocab_size, distractor_prob))(keys)
