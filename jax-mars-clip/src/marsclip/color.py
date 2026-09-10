"""sRGB to CIELAB (D65) and color-affinity bias (Eq. 3)."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

# Bandwidth sigma in Eq. 3.
COLOR_SIGMA = 30.0

# IEC 61966-2-1 sRGB linear RGB -> XYZ (D65).
_SRGB_TO_XYZ = jnp.asarray(
    [
        [0.41239079926595934, 0.35758433938387800, 0.18048078840183430],
        [0.21263900587151027, 0.71516867876775610, 0.07219231536073371],
        [0.01933081871559182, 0.11919477979462598, 0.95053215224966070],
    ],
    dtype=jnp.float32,
)

# CIE D65 white point (2 degree observer), matching the sRGB matrix above.
_XN = 0.95047
_YN = 1.00000
_ZN = 1.08883

# CIE piecewise f(t) threshold: (6/29)^3.
_DELTA = 6.0 / 29.0
_DELTA_CUBED = _DELTA**3
_DELTA_SCALE = 1.0 / (3.0 * _DELTA**2)


def _srgb_to_linear(srgb: Array) -> Array:
    """Linearize sRGB channel values in [0, 1]."""
    return jnp.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def _lab_f(t: Array) -> Array:
    return jnp.where(t > _DELTA_CUBED, jnp.cbrt(t), _DELTA_SCALE * t + 4.0 / 29.0)


def srgb_to_lab(srgb: Array) -> Array:
    """Convert sRGB in [0, 1] to CIELAB (D65, 2 degree).

    Accepts (..., 3) with the last axis RGB, or a CHW image (3, H, W).
    Returns Lab in the same layout.
    """
    chw = srgb.ndim >= 3 and srgb.shape[-3] == 3 and srgb.shape[-1] != 3
    rgb = jnp.moveaxis(srgb, -3, -1) if chw else srgb
    linear = _srgb_to_linear(jnp.clip(rgb, 0.0, 1.0))
    xyz = linear @ _SRGB_TO_XYZ.T
    x = xyz[..., 0] / _XN
    y = xyz[..., 1] / _YN
    z = xyz[..., 2] / _ZN
    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)
    lab = jnp.stack(
        [
            116.0 * fy - 16.0,
            500.0 * (fx - fy),
            200.0 * (fy - fz),
        ],
        axis=-1,
    )
    if chw:
        return jnp.moveaxis(lab, -1, -3)
    return lab


def patch_lab_colors(image_srgb: Array, patch_size: int) -> Array:
    """Average-pool a CHW sRGB image to per-patch Lab colors of shape (Ph, Pw, 3)."""
    if image_srgb.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {image_srgb.shape}")
    lab = srgb_to_lab(image_srgb)
    channels, height, width = lab.shape
    n_h = height // patch_size
    n_w = width // patch_size
    cropped = lab[:, : n_h * patch_size, : n_w * patch_size]
    pooled = cropped.reshape(channels, n_h, patch_size, n_w, patch_size).mean(
        axis=(2, 4)
    )
    return jnp.moveaxis(pooled, 0, -1)


def color_affinity(colors: Array, sigma: float = COLOR_SIGMA) -> Array:
    """Eq. 3: ``B_color(i, j) = exp(-||c_i - c_j||_2^2 / (2 sigma^2))``.

    ``colors`` is (..., P, 3). Returns (..., P, P).
    Identical colors yield affinity 1.
    """
    delta = colors[..., :, None, :] - colors[..., None, :, :]
    dist2 = jnp.sum(delta * delta, axis=-1)
    denom = 2.0 * (sigma**2)
    return jnp.exp(-dist2 / denom)


def color_bias(
    colors: Array,
    sigma: float = COLOR_SIGMA,
    has_cls_token: bool = True,
) -> Array:
    """Color affinity bias for the final KK^T logits.

    Spatial-spatial entries follow Eq. 3. The class token has no spatial
    position, so if ``has_cls_token`` the returned matrix is (P+1, P+1)
    with row 0 and column 0 identically 0 (token 0 is CLS).
    """
    if colors.ndim == 3:
        colors = colors.reshape(-1, colors.shape[-1])
    affinity = color_affinity(colors, sigma)
    if not has_cls_token:
        return affinity
    n_spatial = affinity.shape[-1]
    n_tokens = n_spatial + 1
    leading = affinity.shape[:-2]
    bias = jnp.zeros((*leading, n_tokens, n_tokens), dtype=affinity.dtype)
    return bias.at[..., 1:, 1:].set(affinity)
