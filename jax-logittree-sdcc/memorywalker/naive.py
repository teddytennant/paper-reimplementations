"""The two flattenings the paper argues against (Section 3.3, App. B.4).

Both run a single ordinary causal forward. They differ only in which sequence
they run it on, and they fail in opposite directions.
"""

from __future__ import annotations

import jax.numpy as jnp

from .model import Config, cross_entropy, sequence_logits
from .tree import Rollout


def naive_compressed_loss(params, rollout: Rollout, cfg: Config) -> jnp.ndarray:
    """Pitfall A, ``L^A_SFT`` (Eq. 11): score everything on the final walk.

    Each surviving target is scored under ``p_s(y_t) = H_comp[:t]``, a prefix
    that already carries edits which had not fired when the token was decoded.
    Future compression propagates backward in time, so the model is taught that
    it knew things it had not been told. Tokens that were evicted are gone from
    the sequence and carry no loss at all.
    """
    walk = rollout.compressed_walk()
    ids = jnp.asarray([rollout.tokens[i] for i in walk])
    logits = sequence_logits(params, ids, cfg)
    rows, targets = [], []
    for pos, phys in enumerate(walk):
        if phys >= rollout.response_start and pos > 0:
            rows.append(pos - 1)
            targets.append(rollout.tokens[phys])
    return cross_entropy(logits[jnp.asarray(rows)], jnp.asarray(targets))


def naive_full_loss(params, rollout: Rollout, cfg: Config) -> jnp.ndarray:
    """Pitfall B, ``L^B_SFT`` (Eq. 13): score everything on the physical trace.

    The depth-first traversal of the tree revives every evicted span, so a token
    decoded long after an eviction still attends to the content that eviction
    removed. The model learns to read a context it will never be handed at
    deployment.
    """
    ids = jnp.asarray(rollout.tokens)
    logits = sequence_logits(params, ids, cfg)
    rows = jnp.arange(rollout.response_start - 1, rollout.length - 1)
    targets = jnp.asarray(rollout.tokens[rollout.response_start:])
    return cross_entropy(logits[rows], targets)
