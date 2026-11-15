"""The attention-head influence score of arXiv:2609.05074.

Section 4 gives the directional part, Delta_{k,h} = <w, H_{k,h}>, and the ALTI
proximity of Ferrando et al. (2022) used for relative importance. Section 5.1
combines them into equation (2), S_{k,h} = Delta_{k,h} * C_{k,h} * alpha_k, and
equation (3) aggregates over a chosen subset of layers.

Everything here works on the arrays a `Trace` already holds, so the score never
needs a second forward pass.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

from .model import ModelConfig, Trace, forward, logit_direction


def _l1(x, axis=-1):
    return jnp.abs(x).sum(axis)


def proximity(z: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """ALTI proximity, the unnumbered definition under equation (1).

    proximity(z_i, r) = max(-||z_i - r||_1 + ||r||_1, 0)

    It is a hinge: a component sits at ||r||_1 when it equals the whole state,
    and drops to zero once moving it away from r costs at least as much as r's
    own mass. `z` broadcasts against `r` on all but the last axis.
    """
    return jnp.maximum(_l1(r) - _l1(z - r), 0.0)


def alti_contributions(components: jnp.ndarray, state: jnp.ndarray,
                       axis: int = 0) -> jnp.ndarray:
    """Equation (1): c_i = proximity(z_i, r) / sum_k proximity(z_k, r).

    `axis` says which axis of `components` enumerates the z_i; the last axis is
    always the feature axis the L1 norms run over. If every proximity is zero
    the ratio is undefined and this returns zeros rather than a uniform guess,
    so a degenerate state contributes nothing instead of inventing a share.
    """
    p = proximity(components, state)
    total = p.sum(axis, keepdims=True)
    safe = jnp.where(total > 0, total, 1.0)
    return jnp.where(total > 0, p / safe, 0.0)


def head_contributions(head_writes: jnp.ndarray, attn_out: jnp.ndarray) -> jnp.ndarray:
    """C_{k,h}: head h's share of layer k's attention output, via equation (1).

    Section 5.1 defines it as the head's relative share in the construction of
    the attention output of its layer, so the components are the H_{k,h} and
    the state they are compared against is A_k. Sums to 1 within each layer.

    head_writes is [L, H, D] and attn_out is [L, D]; returns [L, H].
    """
    return alti_contributions(head_writes, attn_out[:, None, :], axis=-1)


def layer_contributions(attn_out: jnp.ndarray, resid_pre_attn: jnp.ndarray,
                        normalize: bool = True) -> jnp.ndarray:
    """C_k: how much layer k's attention output moves the current residual state.

    Section 5.1 gives C_k in words rather than symbols, so this reads it through
    equation (1). The state the attention block produces splits into exactly two
    components, the residual it received and the attention output it added:

        r_mid = r_pre + A_k

    Applying equation (1) to that pair gives A_k's share, which is what "the
    extent to which the attention output modifies the current residual state"
    has to mean if C_k is an ALTI contribution at all. Because A_k - r_mid is
    -r_pre, the share reduces to a closed form in three L1 norms,

        C_k = (||r_mid|| - ||r_pre||) / (2||r_mid|| - ||r_pre|| - ||A_k||)

    and lands in [0, 1] whenever the hinge in `proximity` is inactive, which is
    to say whenever the block grew the L1 norm of the state. A block that shrank
    it gets C_k = 0. That closed form also explains the paper's next sentence,
    that the C_k need normalizing by alpha_k before they can be compared across
    layers: each one is a share of a different state.

    `normalize=False` returns the raw proximity of A_k to r_pre instead. That is
    the other way to read the sentence, and it is worse: A_k is not a piece of
    r_pre, so the hinge in `proximity` clips it to zero for most inputs and the
    whole score collapses.

    attn_out and resid_pre_attn are both [L, D]; returns [L].
    """
    if not normalize:
        return proximity(attn_out, resid_pre_attn)
    resid_mid = resid_pre_attn + attn_out
    pair = jnp.stack([attn_out, resid_pre_attn])
    return alti_contributions(pair, resid_mid, axis=0)[0]


def layer_weights(layer_contribs: jnp.ndarray, layers: Optional[Sequence[int]] = None) -> jnp.ndarray:
    """alpha_k = C_k / sum_j C_j, the normalized layer weight of Section 5.1.

    The paper normalizes over "all considered layers" and then varies the
    considered set in Section 6.2, so the sum runs over `layers` when one is
    given and over every layer otherwise. Entries outside the subset are zero,
    and the rest sum to 1.
    """
    n_layers = layer_contribs.shape[0]
    mask = layer_mask(n_layers, layers, layer_contribs.dtype)
    masked = layer_contribs * mask
    total = masked.sum()
    safe = jnp.where(total > 0, total, 1.0)
    return jnp.where(total > 0, masked / safe, 0.0)


def layer_mask(n_layers: int, layers: Optional[Sequence[int]], dtype=jnp.float32) -> jnp.ndarray:
    if layers is None:
        return jnp.ones((n_layers,), dtype)
    idx = jnp.asarray(list(layers), dtype=jnp.int32)
    return jnp.zeros((n_layers,), dtype).at[idx].set(1.0)


def deltas(head_writes: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Delta_{k,h} = <w, H_{k,h}>, the directional component of Section 4.

    head_writes is [L, H, D]; returns [L, H].
    """
    return head_writes @ w


def influence_scores(delta: jnp.ndarray, c_head: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Equation (2): S_{k,h}(x) = Delta_{k,h}(x) * C_{k,h}(x) * alpha_k(x)."""
    return delta * c_head * alpha[:, None]


class Influence(NamedTuple):
    """Every quantity of Section 5.1 for one input, at the readout position."""

    delta: jnp.ndarray        # [L, H]  Delta_{k,h}
    c_head: jnp.ndarray       # [L, H]  C_{k,h}
    c_layer: jnp.ndarray      # [L]     C_k, each a share of its own layer
    alpha: jnp.ndarray        # [L]     alpha_k over the selected subset
    score: jnp.ndarray        # [L, H]  S_{k,h}
    s_final: jnp.ndarray      # scalar, equation (3)


def influence_from_trace(
    trace: Trace,
    w: jnp.ndarray,
    position: int = 0,
    layers: Optional[Sequence[int]] = None,
) -> Influence:
    """Build every term of the score from a traced forward pass.

    `position` is the sequence position the classifier reads from, so Delta and
    the contributions are all measured on the same residual state the logit
    comes from.
    """
    head_writes = trace.head_writes[:, :, position, :]
    attn_out = trace.attn_out[:, position, :]
    resid_pre = trace.resid_pre_attn[:, position, :]

    delta = deltas(head_writes, w)
    c_head = head_contributions(head_writes, attn_out)
    c_layer = layer_contributions(attn_out, resid_pre)
    alpha = layer_weights(c_layer, layers)
    score = influence_scores(delta, c_head, alpha)
    return Influence(delta, c_head, c_layer, alpha, score, score.sum())


def influence(
    cfg: ModelConfig,
    params,
    tokens: jnp.ndarray,
    layers: Optional[Sequence[int]] = None,
    positive: int = 1,
    negative: int = 0,
    head_mask=None,
) -> Influence:
    """Run the model on one sequence and return its influence scores."""
    trace = forward(cfg, params, tokens, head_mask)
    w = logit_direction(params, positive, negative)
    return influence_from_trace(trace, w, cfg.cls_index, layers)


def batched_influence(cfg, params, tokens, layers=None, positive=1, negative=0):
    """Influence for a batch of sequences, stacked on a leading axis."""
    def one(tok):
        return influence(cfg, params, tok, layers, positive, negative)
    return jax.vmap(one)(tokens)


def s_final(score: jnp.ndarray, layers: Optional[Sequence[int]] = None) -> jnp.ndarray:
    """Equation (3): S_final(x) = sum over the selected layers and all heads.

    Only meaningful when `score` was built with the same layer subset, since
    alpha_k already carries the normalization.
    """
    mask = layer_mask(score.shape[0], layers, score.dtype)
    return (score * mask[:, None]).sum()


def layer_scores(score: jnp.ndarray) -> jnp.ndarray:
    """S_k = sum_h S_{k,h}, the per-layer aggregate of equation (3)."""
    return score.sum(-1)


def signal_split(score: jnp.ndarray):
    """The two signals tracked in Section 6.2.

    Returns (positive signal, benign signal), the sum of positive scores and
    the magnitude of the negative ones.
    """
    return jnp.maximum(score, 0.0).sum(), jnp.maximum(-score, 0.0).sum()


def top_heads(score: jnp.ndarray, n: int, by: str = "abs"):
    """The n dominant heads as (layer, head, score), the ranking of Section 6.3.

    `by` is "abs" for magnitude, "positive" for the strongest pro-positive
    heads, "negative" for the strongest pro-negative ones.
    """
    if by == "abs":
        key = jnp.abs(score)
    elif by == "positive":
        key = score
    elif by == "negative":
        key = -score
    else:
        raise ValueError(f"unknown ranking {by!r}")
    flat = jnp.asarray(key).reshape(-1)
    order = jnp.argsort(-flat)[:n]
    n_heads = score.shape[1]
    return [(int(i) // n_heads, int(i) % n_heads, float(jnp.asarray(score).reshape(-1)[i]))
            for i in order]


TP, TN, FP, FN = 0, 1, 2, 3
CATEGORY_NAMES = {TP: "TP", TN: "TN", FP: "FP", FN: "FN"}


def outcome_categories(preds: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """TP/TN/FP/FN codes for the per-category analysis of Sections 6.1 and 6.3.

    The positive class is the one the score's w points toward, class 1 here,
    matching the paper's convention that Injection is positive.
    """
    preds = jnp.asarray(preds)
    labels = jnp.asarray(labels)
    return jnp.where(
        preds == 1,
        jnp.where(labels == 1, TP, FP),
        jnp.where(labels == 1, FN, TN),
    )


def mean_scores_by_category(scores: jnp.ndarray, categories: jnp.ndarray, category: int) -> jnp.ndarray:
    """Mean S_{k,h} over the examples in one outcome category.

    scores is [N, L, H]; returns [L, H], all zeros if the category is empty.
    """
    sel = (jnp.asarray(categories) == category).astype(scores.dtype)
    count = sel.sum()
    safe = jnp.where(count > 0, count, 1.0)
    return jnp.einsum("n,nlh->lh", sel, scores) / safe
