"""Drift-guided adaptive masking.

Once ``select_mask_layer`` has picked ``l_mask``, the paper thresholds that
layer's attention map and replaces the selected region with gray:

    M(i, j) = 1 if A(i, j) >= Percentile(A, 100 - tau) else 0,     tau = 30,

so pixels at or above the 70th percentile are masked and set to 128.

Two grids, not one. ``A_i`` lives on the ``N`` visual tokens, which for
LLaVA-1.5-7B is 576 tokens laid out 24 by 24, while ``M`` is indexed by pixel
``(i, j)``. The appendix closes the gap: reshape the attention vector to the
``H_g x W_g`` spatial layout of the visual feature map, upsample it to the image
resolution by bilinear interpolation so every pixel carries a continuous
attention weight, and only then take the percentile. The percentile is over the
*upsampled* values, which matters: bilinear interpolation is not
rank-preserving across the whole image, so thresholding the token grid first and
upsampling the binary mask afterwards gives a different mask.

``method="nearest"`` is offered alongside the paper's bilinear. It maps each
token to exactly its own patch of pixels and nothing else, which is the behavior
you want when you are checking that the grid correspondence is right, and it is
what ``tests/test_masking.py`` uses to pin the patch layout down. Bilinear is
the default because it is what the paper ran.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

__all__ = [
    "GRAY",
    "token_grid_shape",
    "upsample_attention",
    "attention_mask",
    "apply_mask",
    "mask_image",
]

GRAY = 128.0
"""Neutral fill value. The appendix ablates black, white and the ImageNet mean
against gray and keeps gray, so 128 is the paper's choice, not a placeholder."""


def token_grid_shape(n: int) -> tuple[int, int]:
    """Infer the square spatial layout of ``n`` visual tokens, e.g. 576 -> (24, 24).

    Models that do not use a square grid should pass ``grid_shape`` explicitly
    to the functions below instead of relying on this.
    """
    side = int(round(math.isqrt(n)))
    if side * side != n:
        raise ValueError(
            f"{n} visual tokens is not a square grid; pass grid_shape explicitly"
        )
    return side, side


def upsample_attention(
    attn: jnp.ndarray,
    image_shape: tuple[int, int],
    grid_shape: tuple[int, int] | None = None,
    method: str = "bilinear",
) -> jnp.ndarray:
    """Lift a token-grid attention map to the pixel grid.

    Args:
        attn: ``[N]`` attention over visual tokens, or an already-reshaped
            ``[H_g, W_g]`` map.
        image_shape: ``(H, W)`` target pixel resolution.
        grid_shape: ``(H_g, W_g)`` token layout. Inferred as square from ``N``
            when ``attn`` is 1-D and this is None.
        method: ``"bilinear"`` (the paper) or ``"nearest"`` (exact patches).

    Returns:
        ``[H, W]`` attention weight per pixel.
    """
    attn = jnp.asarray(attn)
    if attn.ndim == 1:
        if grid_shape is None:
            grid_shape = token_grid_shape(attn.shape[0])
        attn = attn.reshape(grid_shape)
    elif attn.ndim != 2:
        raise ValueError(f"expected [N] or [H_g, W_g] attention, got {attn.shape}")

    height, width = image_shape
    if method == "bilinear":
        return jax.image.resize(attn, (height, width), method="bilinear")
    if method == "nearest":
        grid_h, grid_w = attn.shape
        rows = jnp.floor_divide(jnp.arange(height) * grid_h, height)
        cols = jnp.floor_divide(jnp.arange(width) * grid_w, width)
        return attn[rows[:, None], cols[None, :]]
    raise ValueError(f"unknown upsampling method {method!r}")


def attention_mask(pixel_attn: jnp.ndarray, tau: float = 30.0) -> jnp.ndarray:
    """Threshold a pixel-resolution attention map at the ``100 - tau`` percentile.

    Implements ``M(i, j) = 1[A(i, j) >= Percentile(A, 100 - tau)]``. With the
    paper's ``tau = 30`` the top 30% of pixels by attention are selected.

    The comparison is ``>=``, following the paper. On a map with ties at the
    threshold that selects more than ``tau`` percent of pixels; on a constant
    map it selects all of them. That is the literal reading and it is left
    alone, since real upsampled attention is continuous and does not tie.

    Args:
        pixel_attn: ``[H, W]`` attention per pixel.
        tau: masking percentage, 30 in the paper.

    Returns:
        ``[H, W]`` boolean mask, True where the pixel is to be replaced.
    """
    pixel_attn = jnp.asarray(pixel_attn)
    threshold = jnp.percentile(pixel_attn, 100.0 - tau)
    return pixel_attn >= threshold


def apply_mask(image: jnp.ndarray, mask: jnp.ndarray, fill: float = GRAY) -> jnp.ndarray:
    """Replace masked pixels with ``fill``, leaving the rest untouched.

    Args:
        image: ``[H, W]`` or ``[H, W, C]``. Values are on the 0-255 scale that
            makes 128 a neutral gray; normalize after masking, not before.
        mask: ``[H, W]`` boolean, True where the pixel is replaced.
        fill: replacement value, 128 in the paper.

    Returns:
        Masked image, same shape and dtype-promoted as needed.
    """
    image = jnp.asarray(image)
    mask = jnp.asarray(mask)
    if image.ndim == 3:
        mask = mask[:, :, None]
    return jnp.where(mask, jnp.asarray(fill, image.dtype), image)


def mask_image(
    image: jnp.ndarray,
    attn: jnp.ndarray,
    tau: float = 30.0,
    grid_shape: tuple[int, int] | None = None,
    method: str = "bilinear",
    fill: float = GRAY,
) -> jnp.ndarray:
    """Upsample, threshold and gray out, in one call.

    The full appendix procedure: token-grid attention in, masked image out.
    """
    image = jnp.asarray(image)
    pixel_attn = upsample_attention(
        attn, image.shape[:2], grid_shape=grid_shape, method=method
    )
    return apply_mask(image, attention_mask(pixel_attn, tau=tau), fill=fill)
