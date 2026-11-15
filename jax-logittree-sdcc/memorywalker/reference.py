"""The ground truth: materialize every branch of the tree, one target at a time.

This is ``L*_SFT = -sum_t log P_theta(y_t | p_t(y_t))`` (Eq. 11), computed the
slowest possible way. Each response token gets its own forward pass over its own
live view, so there is no packing, no shared trunk and no mask trickery that
could hide a bug. Everything else in the repo is checked against this.
"""

from __future__ import annotations

import jax.numpy as jnp

from .model import Config, cross_entropy, sequence_logits
from .tree import Rollout


def per_token_logits(params, rollout: Rollout, cfg: Config) -> dict[int, jnp.ndarray]:
    """``P_theta(. | p_t(y_t))`` at every loss-carrying position, one forward each."""
    out = {}
    for t in rollout.response_positions():
        prefix = rollout.live_view(t)
        ids = jnp.asarray([rollout.tokens[i] for i in prefix])
        out[t] = sequence_logits(params, ids, cfg)[-1]
    return out


def materialized_tree_loss(params, rollout: Rollout, cfg: Config) -> jnp.ndarray:
    """Summed cross-entropy of every leaf under the prefix it was decoded from."""
    total = jnp.zeros((), dtype=jnp.float64)
    for t, logits in per_token_logits(params, rollout, cfg).items():
        total = total + cross_entropy(logits[None], jnp.asarray([rollout.tokens[t]]))
    return total
