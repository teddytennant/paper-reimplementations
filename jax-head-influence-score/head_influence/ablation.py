"""Head ablation, the causal check of Section 6.4.

Ablating a head means deleting its write H_{k,h} from the residual stream and
recomputing the logit, which is what `head_mask` sets up. Note that this costs
more than Delta_{k,h}: every block above the ablated head now reads a different
residual state, so its attention pattern and its MLP move too.
`direct_path_logit_diff` is the other ablation, the one that isolates the term
Delta actually measures.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp

from .model import ModelConfig, batched_logits, forward, logit_direction


def head_mask(cfg: ModelConfig, ablated: Iterable[Tuple[int, int]] = (), dtype=jnp.float32):
    """A [L, H] mask with zeros at the (layer, head) pairs to delete."""
    mask = jnp.ones((cfg.n_layers, cfg.n_heads), dtype)
    for layer, head in ablated:
        mask = mask.at[layer, head].set(0.0)
    return mask


def logit_diff(cfg, params, tokens, mask=None, positive=1, negative=0):
    """Batched logit difference, the quantity the score is written against."""
    logits = batched_logits(cfg, params, tokens, mask)
    return logits[:, positive] - logits[:, negative]


def accuracy(cfg, params, tokens, labels, mask=None):
    preds = batched_logits(cfg, params, tokens, mask).argmax(-1)
    return (preds == labels).mean()


def accuracy_for_masks(cfg, params, tokens, labels, masks):
    """Accuracy for a stack of [N, L, H] masks, one forward sweep per mask."""
    def one(mask):
        return accuracy(cfg, params, tokens, labels, mask)
    return jax.lax.map(one, masks)


def all_subset_masks(cfg: ModelConfig, size: int, dtype=jnp.float32):
    """Every mask that deletes exactly `size` heads, plus the head tuples.

    Enumerating the whole set makes the Section 6.4 comparison deterministic:
    the top-|S| set can be scored against the full distribution of ablations of
    the same size instead of against one random draw.
    """
    heads = [(l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    subsets = list(itertools.combinations(heads, size))
    masks = jnp.stack([head_mask(cfg, s, dtype) for s in subsets])
    return masks, subsets


def direct_path_logit_diff(cfg, params, tokens, ablated=(), positive=1, negative=0):
    """Logit difference with the ablated heads removed from the direct path only.

    Delta_{k,h} = <w, H_{k,h}> is a direct-path quantity: it reads the head's
    write off the final residual state and says nothing about how the rest of
    the network would react to losing it. Removing the write from r^(L) while
    holding every downstream component at its clean value is the ablation that
    matches that definition, and it moves the logit difference by exactly
    -Delta_{k,h}.

    Deleting the write and rerunning the model instead, which is what
    `head_mask` does and what Section 6.4 measures, also perturbs every later
    block, so the two differ by the downstream reaction.
    """
    trace = forward(cfg, params, tokens)
    w = logit_direction(params, positive, negative)
    removed = jnp.zeros_like(w)
    for layer, head in ablated:
        removed = removed + trace.head_writes[layer, head, cfg.cls_index]
    return trace.logits[positive] - trace.logits[negative] - w @ removed


def direct_path_accuracy(cfg, params, tokens, labels, ablated=()):
    """Accuracy when the ablated heads lose only their direct path to the logit.

    Same prediction rule as `accuracy`, since for two classes the argmax is the
    sign of the logit difference.
    """
    diffs = jax.vmap(lambda t: direct_path_logit_diff(cfg, params, t, ablated))(tokens)
    return ((diffs > 0).astype(jnp.int32) == labels).mean()
