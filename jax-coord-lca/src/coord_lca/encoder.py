"""Toy downsampled encoder (not a foundation ViT).

Downsample the image, then a linear patch embedding to a coarse token grid
``(H_e, W_e, D)``. This stubs the paper's ViT foundation encoder ``E``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from coord_lca.patch import init_linear, linear


def avg_pool(image: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Integer-factor average pooling, ``(H, W, C) -> (H/f, W/f, C)``."""
    if factor == 1:
        return image
    height, width, channels = image.shape
    dh = height // factor
    dw = width // factor
    cropped = image[: dh * factor, : dw * factor, :]
    return cropped.reshape(dh, factor, dw, factor, channels).mean(axis=(1, 3))


def spatial_size(
    height: int,
    width: int,
    *,
    downsample: int,
    patch_size: int,
) -> tuple[int, int]:
    return height // (downsample * patch_size), width // (downsample * patch_size)


def init_encoder(
    key: jax.Array,
    *,
    in_channels: int = 3,
    d_model: int = 32,
    patch_size: int = 4,
) -> dict:
    in_dim = patch_size * patch_size * in_channels
    k_lin, k_pos = jax.random.split(key)
    return {
        "patch": init_linear(k_lin, in_dim, d_model),
        "pos_scale": jax.random.normal(k_pos, (d_model,), dtype=jnp.float32) * 0.02,
    }


def _patchify(image: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    height, width, channels = image.shape
    nh = height // patch_size
    nw = width // patch_size
    cropped = image[: nh * patch_size, : nw * patch_size, :]
    patches = cropped.reshape(nh, patch_size, nw, patch_size, channels)
    patches = jnp.transpose(patches, (0, 2, 1, 3, 4))
    return patches.reshape(nh, nw, patch_size * patch_size * channels)


def _sinusoidal_posemb(height: int, width: int, dim: int) -> jnp.ndarray:
    """Fixed 2D sinusoidal positional encoding, shape ``(H, W, dim)``."""
    y = jnp.arange(height, dtype=jnp.float32)
    x = jnp.arange(width, dtype=jnp.float32)
    gy, gx = jnp.meshgrid(y, x, indexing="ij")
    half = dim // 4
    if half == 0:
        return jnp.zeros((height, width, dim), dtype=jnp.float32)
    div = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -(jnp.log(10000.0) / max(half, 1)))
    pe_x = jnp.stack(
        [jnp.sin(gx[..., None] * div), jnp.cos(gx[..., None] * div)], axis=-1
    ).reshape(height, width, 2 * half)
    pe_y = jnp.stack(
        [jnp.sin(gy[..., None] * div), jnp.cos(gy[..., None] * div)], axis=-1
    ).reshape(height, width, 2 * half)
    pe = jnp.concatenate([pe_x, pe_y], axis=-1)
    if pe.shape[-1] < dim:
        pe = jnp.pad(pe, ((0, 0), (0, 0), (0, dim - pe.shape[-1])))
    return pe[..., :dim]


def encode(
    params: dict,
    image: jnp.ndarray,
    *,
    downsample: int = 2,
    patch_size: int = 4,
) -> jnp.ndarray:
    """Encode a full-res image to a coarse token grid ``(H_e, W_e, D)``."""
    down = avg_pool(image, downsample)
    patches = _patchify(down, patch_size)
    tokens = linear(params["patch"], patches)
    height, width, dim = tokens.shape
    pos = _sinusoidal_posemb(height, width, dim) * params["pos_scale"]
    return tokens + pos
