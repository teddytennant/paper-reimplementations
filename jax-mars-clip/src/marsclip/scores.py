"""Zero-shot cosine scores and class assignment (Eq. 4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray

_EPS = 1e-8


def _l2_normalize(x: Array, axis: int = -1) -> Array:
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, _EPS)


def cosine_scores(f_refined: Array, f_text: Array) -> Array:
    """Eq. 4: ``S_{i,j,k} = cosine(F_refined[i,j], F_text[k])``.

    ``f_refined`` is (..., H, W, D) and ``f_text`` is (K, D). Returns
    (..., H, W, K).
    """
    image = _l2_normalize(f_refined, axis=-1)
    text = _l2_normalize(f_text, axis=-1)
    return jnp.einsum("...hwd,kd->...hwk", image, text)


def predict_mask(scores: Array, target_hw: tuple[int, int]) -> Array:
    """Bilinear-upsample score maps to the original image size, then argmax over k.

    Does not apply PAMR. ``scores`` is (..., H, W, K); returns integer labels
    of shape (..., H_out, W_out).
    """
    out_h, out_w = target_hw
    leading = scores.shape[:-3]
    n_classes = scores.shape[-1]
    upsampled = jax.image.resize(
        scores,
        (*leading, out_h, out_w, n_classes),
        method="linear",
    )
    return jnp.argmax(upsampled, axis=-1)
