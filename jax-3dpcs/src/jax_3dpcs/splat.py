"""Min-depth Chebyshev splatting for 3DPCS (PRI-Net eq. 8).

Paper eq. 8 fills a dense depth map by assigning each projected point's depth
z^c to every pixel within an L_inf (Chebyshev) disk of radius R, keeping the
nearest depth where disks overlap:

    M_i(u^v) = min_u { z^c(u) }  s.t.  dist_inf(u^v, u) <= R

Empty pixels are not defined in the paper. This kernel records occupancy O
(1 where at least one splat landed) and returns D_final = D_raw with empty
pixels replaced by 0 rather than +inf.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_3dpcs.camera import project_P, project_pinhole

Array = jnp.ndarray


class SplatMaps(NamedTuple):
    """Depth and occupancy maps from 3DPCS.

    D_raw uses +inf on empty pixels (true min over an empty set).
    O is 1 where at least one splat landed, else 0.
    D_final is D_raw with empty pixels set to 0 (usable depth map M).
    """

    D_raw: Array
    O: Array
    D_final: Array


def chebyshev_offsets(radius: int) -> tuple[Array, Array]:
    """Integer pixel offsets (du, dv) with L_inf <= radius, shape (K,) each."""
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    offs = jnp.arange(-r, r + 1, dtype=jnp.int32)
    du, dv = jnp.meshgrid(offs, offs, indexing="xy")
    return du.reshape(-1), dv.reshape(-1)


def _scatter_min_flat(size: int, indices: Array, values: Array) -> Array:
    """1-D min-reduce scatter. Out-of-range indices are dropped."""
    operand = jnp.full((size,), jnp.inf, dtype=values.dtype)
    scatter_indices = indices.astype(jnp.int32).reshape(-1, 1)
    updates = values.reshape(-1)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return jax.lax.scatter_min(
        operand,
        scatter_indices,
        updates,
        dnums,
        mode="drop",
    )


def splat_projected(
    uv: Array,
    z_c: Array,
    valid: Array,
    height: int,
    width: int,
    radius: int,
) -> SplatMaps:
    """Min-depth Chebyshev splat from already-projected points (eq. 8).

    Projected (u, v) are rounded to the nearest pixel; the inclusive integer
    square of radius R around that pixel is filled. Points behind the camera
    (valid=False) or whose rounded pixel lies outside the image are ignored.
    Splat pixels that fall off the image border are dropped.
    """
    uv = jnp.asarray(uv)
    z_c = jnp.asarray(z_c)
    valid = jnp.asarray(valid)
    height = int(height)
    width = int(width)
    du, dv = chebyshev_offsets(radius)

    u0 = jnp.round(uv[:, 0]).astype(jnp.int32)
    v0 = jnp.round(uv[:, 1]).astype(jnp.int32)
    in_image = (u0 >= 0) & (u0 < width) & (v0 >= 0) & (v0 < height)
    keep = valid & in_image

    uu = u0[:, None] + du[None, :]
    vv = v0[:, None] + dv[None, :]
    zz = jnp.broadcast_to(z_c[:, None], uu.shape)
    keep_pix = (
        keep[:, None]
        & (uu >= 0)
        & (uu < width)
        & (vv >= 0)
        & (vv < height)
    )
    # Dropped indices use -1 so scatter_min(mode='drop') ignores them.
    flat_idx = jnp.where(keep_pix, vv * width + uu, -1).reshape(-1)
    flat_z = zz.reshape(-1).astype(jnp.float32)

    D_raw = _scatter_min_flat(height * width, flat_idx, flat_z).reshape(
        height, width
    )
    O = jnp.isfinite(D_raw).astype(jnp.float32)
    D_final = jnp.where(O > 0, D_raw, jnp.zeros_like(D_raw))
    return SplatMaps(D_raw=D_raw, O=O, D_final=D_final)


def splat_3dpcs(
    points: Array,
    *,
    height: int,
    width: int,
    radius: int,
    K: Array | None = None,
    R: Array | None = None,
    t: Array | None = None,
    P: Array | None = None,
) -> SplatMaps:
    """3DPCS: pinhole project (eq. 7) then min-depth L_inf splat (eq. 8).

    Provide either K (and optional R, t; defaults identity / zero) or a 3x4 P.
    """
    points = jnp.asarray(points)
    if points.size == 0:
        points = points.reshape(0, 3)
    if P is not None:
        uv, z_c, valid = project_P(points, P)
    else:
        if K is None:
            raise ValueError("splat_3dpcs requires K or P")
        uv, z_c, valid = project_pinhole(points, K, R=R, t=t)
    return splat_projected(uv, z_c, valid, height, width, radius)
