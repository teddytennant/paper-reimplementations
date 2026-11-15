"""Run Gride on a synthetic 3D Gaussian embedded in R^10."""

import sys

import jax
import jax.numpy as jnp

from gride_id import estimate_id


def main() -> int:
    key_data, key_est = jax.random.split(jax.random.PRNGKey(0))
    latent = jax.random.normal(key_data, (400, 3), dtype=jnp.float64)
    cloud = jnp.pad(latent, ((0, 0), (0, 7)))
    result = estimate_id(cloud, key=key_est)
    d_hat = result["d_hat"]
    d_dist = result["d_dist"]
    print(f"d_hat {d_hat}")
    print(f"d_dist {d_dist}")
    if d_hat in {1, 2, 3, 4, 5}:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
