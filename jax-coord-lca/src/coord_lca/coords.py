"""Coordinate embeddings and image-to-encoder-grid mapping.

Coordinate convention
---------------------
Pixel coordinates ``(x, y)`` live on a continuous canvas with origin at the
top-left of the image, ``x`` rightward and ``y`` downward. Valid range is
``x in [0, W]``, ``y in [0, H]``, so image corners map to encoder-grid corners.
Integer pixel ``(row i, col j)`` occupies ``[j, j+1] x [i, i+1]`` and has
center ``(j + 0.5, i + 0.5)``.

Normalized coordinates used by the Fourier embedding are ``(x / W, y / H)``
in ``[0, 1]``.

Encoder-grid mapping (paper eq. 7)
----------------------------------
Let ``s_W = W / W_down``, ``s_H = H / H_down`` and ``p_e`` the encoder patch
size. Then

    (x_down, y_down) = (x / (s_W * p_e), y / (s_H * p_e))

which is identical to ``(x * W_e / W, y * H_e / H)`` with
``W_e = W_down / p_e``, ``H_e = H_down / p_e``.

Token ``(i, j)`` (row ``i``, column ``j``) occupies encoder-grid rectangle
``[j, j+1] x [i, i+1]``. Its center ``(j + 0.5, i + 0.5)`` is the image of the
center of that encoder cell.
"""

from __future__ import annotations

import jax.numpy as jnp

# Each frequency contributes sin/cos on x and on y.
FOURIER_OUT_MULTIPLIER = 4


def fourier_frequencies(n_freq: int) -> jnp.ndarray:
    """NeRF-style frequencies ``2^k * pi`` for ``k = 0 .. n_freq-1``."""
    k = jnp.arange(n_freq, dtype=jnp.float32)
    return (2.0**k) * jnp.pi


def fourier_embed(xy_norm: jnp.ndarray, n_freq: int = 6) -> jnp.ndarray:
    """Fourier / sinusoidal embedding of normalized coordinates.

    Parameters
    ----------
    xy_norm:
        ``(..., 2)`` array of ``(x, y)`` in ``[0, 1]``.
    n_freq:
        Number of frequency octaves.

    Returns
    -------
    ``(..., 4 * n_freq)`` embedding

        ``[sin(2^k pi x), cos(2^k pi x), sin(2^k pi y), cos(2^k pi y)]_k``

    concatenated in that order (sin_x, cos_x, sin_y, cos_y), each of length
    ``n_freq``. Trigonometric features lie in ``[-1, 1]``.
    """
    xy_norm = jnp.asarray(xy_norm, dtype=jnp.float32)
    freqs = fourier_frequencies(n_freq)
    x = xy_norm[..., 0:1] * freqs
    y = xy_norm[..., 1:2] * freqs
    return jnp.concatenate(
        [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)],
        axis=-1,
    )


def normalize_coords(xy: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
    """Map pixel ``(x, y)`` on ``[0, W] x [0, H]`` to ``(x/W, y/H)``."""
    xy = jnp.asarray(xy, dtype=jnp.float32)
    scale = jnp.array([width, height], dtype=jnp.float32)
    return xy / scale


def image_to_encoder_grid(
    xy: jnp.ndarray,
    *,
    height: int,
    width: int,
    encoder_height: int | None = None,
    encoder_width: int | None = None,
    downsample: int | None = None,
    encoder_patch_size: int | None = None,
) -> jnp.ndarray:
    """Map image-space ``(x, y)`` to encoder-grid coordinates (eq. 7).

    Provide either ``(encoder_height, encoder_width)`` or
    ``(downsample, encoder_patch_size)``. The latter uses

        ``s_W = downsample``, ``s_H = downsample``
        (i.e. ``W_down = W / downsample``)

    so ``(x_down, y_down) = (x / (s_W * p_e), y / (s_H * p_e))``.
    """
    xy = jnp.asarray(xy, dtype=jnp.float32)
    if encoder_height is None or encoder_width is None:
        if downsample is None or encoder_patch_size is None:
            raise ValueError(
                "Provide encoder_height/encoder_width or downsample and "
                "encoder_patch_size."
            )
        # eq. 7 with s_W = W / W_down = downsample, s_H = downsample
        s_w = float(downsample)
        s_h = float(downsample)
        p_e = float(encoder_patch_size)
        x_down = xy[..., 0] / (s_w * p_e)
        y_down = xy[..., 1] / (s_h * p_e)
        return jnp.stack([x_down, y_down], axis=-1)

    x_down = xy[..., 0] * (encoder_width / float(width))
    y_down = xy[..., 1] * (encoder_height / float(height))
    return jnp.stack([x_down, y_down], axis=-1)


def dense_query_coords(height: int, width: int, *, centers: bool = True) -> jnp.ndarray:
    """Grid of pixel coordinates with shape ``(H, W, 2)`` as ``(x, y)``.

    If ``centers`` is true, integer pixel ``(i, j)`` is queried at its center
    ``(j + 0.5, i + 0.5)``; otherwise at its top-left corner ``(j, i)``.
    """
    ys = jnp.arange(height, dtype=jnp.float32)
    xs = jnp.arange(width, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    if centers:
        grid_x = grid_x + 0.5
        grid_y = grid_y + 0.5
    return jnp.stack([grid_x, grid_y], axis=-1)
