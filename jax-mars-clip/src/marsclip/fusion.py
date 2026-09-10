"""Multi-resolution feature extraction and fusion (Eq. 1).

Given I in R^{3 x H x W}:
  1. Reflective-pad so H, W are multiples of S_crop.
  2. Split into an N x M grid of S_crop x S_crop regions; resize each to K x K.
  3. Encode each crop to F_local^{(n,m)} and stitch by grid position into F_local.
  4. Encode the full image resized to K x K as F_global.
  5. F_refined = alpha * Up(F_global) + (1 - alpha) * F_local
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from marsclip.constants import ALPHA, ENCODER_SIZE, S_CROP

Array = jnp.ndarray


def pad_to_multiple(image: Array, s_crop: int = S_CROP) -> Array:
    """Reflective-pad a CHW image so H and W are multiples of ``s_crop``.

    Padding is applied at the bottom and right. If a side is already a
    multiple of ``s_crop``, that side is unchanged.
    """
    if image.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {image.shape}")
    _c, height, width = image.shape
    pad_h = (s_crop - height % s_crop) % s_crop
    pad_w = (s_crop - width % s_crop) % s_crop
    if pad_h == 0 and pad_w == 0:
        return image
    return jnp.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")


def grid_shape(padded_hw: tuple[int, int], s_crop: int = S_CROP) -> tuple[int, int]:
    """Return (N, M) so that N * S_crop and M * S_crop equal the padded size."""
    height, width = padded_hw
    if height % s_crop != 0 or width % s_crop != 0:
        raise ValueError(
            f"padded size {(height, width)} is not a multiple of s_crop={s_crop}"
        )
    return height // s_crop, width // s_crop


def grid_crops(padded: Array, s_crop: int = S_CROP) -> Array:
    """Split a padded CHW image into an (N, M, C, S_crop, S_crop) crop grid."""
    if padded.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {padded.shape}")
    channels, height, width = padded.shape
    n_rows, n_cols = grid_shape((height, width), s_crop)
    crops = padded.reshape(channels, n_rows, s_crop, n_cols, s_crop)
    return jnp.transpose(crops, (1, 3, 0, 2, 4))


def resize_chw(image: Array, size: int) -> Array:
    """Bilinear-resize a CHW image (or a stack with leading axes) to ``size x size``.

    ``image`` has shape (..., C, H, W). Channels stay in the same axis.
    """
    if image.ndim < 3:
        raise ValueError(f"expected at least CHW, got shape {image.shape}")
    channels = image.shape[-3]
    leading = image.shape[:-3]
    hwc = jnp.moveaxis(image, -3, -1)
    resized = jax.image.resize(
        hwc,
        (*leading, size, size, channels),
        method="linear",
    )
    return jnp.moveaxis(resized, -1, -3)


def upsample_features(features: Array, target_hw: tuple[int, int]) -> Array:
    """Bilinear-upsample spatial feature maps to ``target_hw``.

    ``features`` has shape (..., H, W, D).
    """
    if features.ndim < 3:
        raise ValueError(f"expected (..., H, W, D), got shape {features.shape}")
    out_h, out_w = target_hw
    leading = features.shape[:-3]
    depth = features.shape[-1]
    return jax.image.resize(
        features,
        (*leading, out_h, out_w, depth),
        method="linear",
    )


def fuse_features(
    f_global: Array,
    f_local: Array,
    alpha: float = ALPHA,
) -> Array:
    """Eq. 1: ``F_refined = alpha * Up(F_global) + (1 - alpha) * F_local``.

    Both maps are (..., H, W, D). ``Up`` bilinear-resizes ``f_global`` to the
    spatial size of ``f_local``.
    """
    target_hw = (f_local.shape[-3], f_local.shape[-2])
    up_global = upsample_features(f_global, target_hw)
    return alpha * up_global + (1.0 - alpha) * f_local


def reassemble_local_features(local_maps: Array) -> Array:
    """Stitch per-crop maps (N, M, h, w, D) into one (N*h, M*w, D) map."""
    if local_maps.ndim != 5:
        raise ValueError(
            f"expected (N, M, h, w, D) local maps, got shape {local_maps.shape}"
        )
    n_rows, n_cols, feat_h, feat_w, depth = local_maps.shape
    tiled = jnp.transpose(local_maps, (0, 2, 1, 3, 4))
    return tiled.reshape(n_rows * feat_h, n_cols * feat_w, depth)


def encode_multiresolution(
    image: Array,
    encoder: Callable[[Array], Array],
    s_crop: int = S_CROP,
    encoder_size: int = ENCODER_SIZE,
    alpha: float = ALPHA,
) -> Array:
    """Pad, grid, encode local crops and the full image, then fuse (Eq. 1).

    Parameters
    ----------
    image:
        CHW float array in R^{3 x H x W}.
    encoder:
        Callable mapping a CHW crop of size K x K to a spatial feature map
        (h, w, D). Applied independently to each local crop and to the
        globally resized image.
    """
    padded = pad_to_multiple(image, s_crop)
    crops = grid_crops(padded, s_crop)
    resized_crops = resize_chw(crops, encoder_size)
    local_maps = jax.vmap(jax.vmap(encoder))(resized_crops)
    f_local = reassemble_local_features(local_maps)
    global_input = resize_chw(image, encoder_size)
    f_global = encoder(global_input)
    return fuse_features(f_global, f_local, alpha)
