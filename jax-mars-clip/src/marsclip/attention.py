"""Structure-aware KK^T attention (Eq. 2) without residual or FFN."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def key_key_logits(k_final: Array) -> Array:
    """KK^T / sqrt(d) logits from final-layer keys of shape (..., T, d)."""
    depth = k_final.shape[-1]
    scale = jnp.sqrt(jnp.asarray(depth, dtype=k_final.dtype))
    return jnp.matmul(k_final, jnp.moveaxis(k_final, -1, -2)) / scale


def combine_biases(
    b_int: Array | None = None,
    b_color: Array | None = None,
) -> Array | None:
    """B_total = B_int + B_color. Either term may be omitted (treated as 0)."""
    if b_int is None and b_color is None:
        return None
    if b_int is None:
        return b_color
    if b_color is None:
        return b_int
    return b_int + b_color


def structure_aware_attention(
    k_final: Array,
    b_int: Array | None = None,
    b_color: Array | None = None,
    b_total: Array | None = None,
) -> Array:
    """Eq. 2: ``A_final = softmax(KK^T / sqrt(d) + B_total)``.

    ``B_total`` is ``B_int + B_color`` when not given explicitly. ``B_int`` is
    the intermediate-layer attention map (post-softmax), added to the logits.
    Color bias is not applied to the class token by construction of ``b_color``.
    """
    logits = key_key_logits(k_final)
    if b_total is None:
        b_total = combine_biases(b_int, b_color)
    if b_total is not None:
        logits = logits + b_total
    return jax.nn.softmax(logits, axis=-1)


def linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    """Affine map ``x @ weight (+ bias)`` with weight shape (in, out)."""
    out = jnp.matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def refined_attention_block(
    k_final: Array,
    v_final: Array,
    weight: Array,
    bias: Array | None = None,
    b_int: Array | None = None,
    b_color: Array | None = None,
    b_total: Array | None = None,
) -> Array:
    """Final refined block: ``O_final = Linear(A_final @ V_final)``.

    No residual connection and no feed-forward network (ClearCLIP).
    """
    attn = structure_aware_attention(
        k_final, b_int=b_int, b_color=b_color, b_total=b_total
    )
    aggregated = jnp.matmul(attn, v_final)
    return linear(aggregated, weight, bias)
