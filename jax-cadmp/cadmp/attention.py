"""Head-averaged cross-modal attention, eq. (1) of the paper.

The paper's setup: for the query position whose logits predict the target token
``y``, layer ``i`` and head ``h`` assign weight ``Attn_{i,h}(y, v_m)`` to the
``m``-th visual token. Eq. (1) averages that over the ``H`` heads,

    a_i^(m) = (1/H) * sum_h Attn_{i,h}(y, v_m),

giving one vector ``A_i`` per layer over the ``N`` visual tokens. Stacking the
layers gives ``A = {A_1, ..., A_L}``.

Normalization. The paper says only "after normalization, A_i can be viewed as a
discrete probability distribution over the visual-token index space", without
writing the operation down. The head-averaged weights do not sum to one on their
own: a decoder-layer softmax runs over the whole prefix, so the visual tokens
carry only part of the mass and the rest sits on the system prompt, the text
prompt and the generated prefix. So the normalization has to be a restriction of
the softmax to the visual subset,

    p_i(x) = a_i^(x) / sum_m a_i^(m),

which is what ``normalize`` does. This is the only reading that makes the KL in
the next section a divergence between distributions over the visual-token index
space. Dividing by ``N``, or a fresh softmax over the head-averaged weights,
would both change the geometry of the drift profile.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["head_average", "normalize", "attention_distributions"]


def head_average(attn: jnp.ndarray) -> jnp.ndarray:
    """Average cross-modal attention over heads, eq. (1).

    Args:
        attn: ``[L, H, N]`` attention from the target-token query position to
            the ``N`` visual tokens, for ``L`` decoder layers and ``H`` heads.

    Returns:
        ``[L, N]`` head-averaged weights, not yet normalized.
    """
    attn = jnp.asarray(attn)
    if attn.ndim != 3:
        raise ValueError(f"expected attention of shape [L, H, N], got {attn.shape}")
    return jnp.mean(attn, axis=1)


def normalize(a: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Restrict head-averaged attention to a distribution over visual tokens.

    Divides each layer's vector by its sum over the ``N`` visual tokens, so each
    row sums to 1. ``eps`` guards a layer that put no mass at all on the visual
    tokens; it is a denominator guard only and does not shift the result for any
    row with a nonzero sum at float64, and by ~1e-12 relative at float32.

    Args:
        a: ``[..., N]`` non-negative weights.
        eps: floor on the row sum.

    Returns:
        Array of the same shape whose last axis sums to 1.
    """
    a = jnp.asarray(a)
    total = jnp.sum(a, axis=-1, keepdims=True)
    return a / jnp.maximum(total, eps)


def attention_distributions(attn: jnp.ndarray) -> jnp.ndarray:
    """``head_average`` then ``normalize``: raw ``[L, H, N]`` to ``[L, N]``.

    The result is the layer-wise sequence ``A = {A_1, ..., A_L}`` that the drift
    profile is computed from.
    """
    return normalize(head_average(attn))
