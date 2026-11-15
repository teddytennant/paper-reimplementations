"""Synthetic cube splat at 64x64. Prints occupancy fraction and depth range."""

from __future__ import annotations

import jax.numpy as jnp

from jax_3dpcs.splat import splat_3dpcs


def main() -> None:
    H = W = 64
    radius = 2
    K = jnp.array(
        [[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    # Cube [-1, 1]^3, yawed so near and far silhouette pixels keep different depths.
    lin = jnp.linspace(-1.0, 1.0, 9)
    xx, yy, zz = jnp.meshgrid(lin, lin, lin, indexing="ij")
    obj = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    yaw = 0.6
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    rot = jnp.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=jnp.float32
    )
    points = obj @ rot.T + jnp.array([0.0, 0.0, 5.0], dtype=jnp.float32)

    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=radius)
    occ = maps.O
    occupied = occ > 0
    frac = float(occ.mean())
    d = maps.D_final
    d_min = float(jnp.min(jnp.where(occupied, d, jnp.inf)))
    d_max = float(jnp.max(jnp.where(occupied, d, -jnp.inf)))
    print(f"n_points={points.shape[0]} map={H}x{W} radius={radius}")
    print(f"occupancy_fraction={frac:.6f}")
    print(f"depth_min={d_min:.6f} depth_max={d_max:.6f}")
    print(f"occupied_pixels={int(occupied.sum())} / {H * W}")


if __name__ == "__main__":
    main()
