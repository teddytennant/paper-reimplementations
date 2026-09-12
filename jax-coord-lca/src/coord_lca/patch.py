"""High-resolution local patch extract and MLP embedder.

Paper: given query ``(x, y)``, extract an 8x8 patch ``P_{x,y}`` whose top-left
corner is aligned with that position, flatten, and map with MLP ``Psi``:

    f_p = Psi(P_{x,y}),   Psi : R^{3 p^2} -> R^{d_p}
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from coord_lca.coords import normalize_coords, fourier_embed


def init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict:
    w_key, b_key = jax.random.split(key)
    w = jax.random.normal(w_key, (in_dim, out_dim), dtype=jnp.float32) * (
        1.0 / jnp.sqrt(in_dim)
    )
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"W": w, "b": b}


def linear(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["W"] + params["b"]


def init_mlp(key: jax.Array, sizes: tuple[int, ...]) -> list:
    keys = jax.random.split(key, len(sizes) - 1)
    return [init_linear(k, din, dout) for k, din, dout in zip(keys, sizes[:-1], sizes[1:])]


def mlp(params: list, x: jnp.ndarray, *, last_linear: bool = True) -> jnp.ndarray:
    last = len(params) - 1
    for i, layer in enumerate(params):
        x = linear(layer, x)
        if i != last or not last_linear:
            x = jax.nn.relu(x)
    return x


def init_patch_embed(
    key: jax.Array,
    *,
    patch_size: int = 8,
    in_channels: int = 3,
    d_patch: int = 32,
    hidden: int | None = None,
) -> list:
    """Two-layer MLP ``3 p^2 -> hidden -> d_patch``."""
    in_dim = patch_size * patch_size * in_channels
    if hidden is None:
        hidden = max(d_patch, in_dim // 2)
    return init_mlp(key, (in_dim, hidden, d_patch))


def extract_patches(
    image: jnp.ndarray,
    xy: jnp.ndarray,
    patch_size: int = 8,
) -> jnp.ndarray:
    """Extract top-left-aligned ``p x p`` patches at pixel coordinates ``xy``.

    Parameters
    ----------
    image:
        ``(H, W, C)`` full-resolution image.
    xy:
        ``(N, 2)`` pixel coordinates ``(x, y)``. The patch top-left is
        ``(floor(x), floor(y))``. Out-of-bounds samples are zero-padded.
    patch_size:
        Spatial patch size ``p`` (paper uses 8).

    Returns
    -------
    ``(N, p, p, C)`` patches.
    """
    image = jnp.asarray(image)
    xy = jnp.asarray(xy)
    height, width, channels = image.shape
    pad = int(patch_size)
    padded = jnp.pad(image, ((pad, pad), (pad, pad), (0, 0)))
    x0 = jnp.floor(xy[..., 0]).astype(jnp.int32)
    y0 = jnp.floor(xy[..., 1]).astype(jnp.int32)
    max_y = height + 2 * pad - patch_size
    max_x = width + 2 * pad - patch_size

    def one(x_i, y_i):
        sy = jnp.clip(y_i + pad, 0, max_y)
        sx = jnp.clip(x_i + pad, 0, max_x)
        return jax.lax.dynamic_slice(
            padded, (sy, sx, 0), (patch_size, patch_size, channels)
        )

    return jax.vmap(one)(x0, y0)


def embed_patches(params: list, patches: jnp.ndarray) -> jnp.ndarray:
    """Flatten ``(N, p, p, C)`` patches and apply ``Psi``."""
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    return mlp(params, flat)


def patch_and_coord_features(
    patch_params: list,
    image: jnp.ndarray,
    xy: jnp.ndarray,
    *,
    patch_size: int = 8,
    n_freq: int = 6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(f_p, f_c)`` for queries ``xy`` on ``image``."""
    height, width = image.shape[:2]
    patches = extract_patches(image, xy, patch_size=patch_size)
    f_p = embed_patches(patch_params, patches)
    xy_norm = normalize_coords(xy, height, width)
    f_c = fourier_embed(xy_norm, n_freq=n_freq)
    return f_p, f_c
