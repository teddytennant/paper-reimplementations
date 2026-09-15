"""Tiny DCTS inverse-problem smoke run (CPU)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dcts.denoisers import gaussian_ve_denoiser
from dcts.operators import gaussian_matrix
from dcts.sampler import dcts
from dcts.schedule import edm_sigmas


def main() -> None:
    key = jax.random.PRNGKey(0)
    d = 32
    key_z, key_a, key_n, key_s = jax.random.split(key, 4)
    x0 = jax.random.normal(key_z, (d,))
    a = gaussian_matrix(key_a, m=32, d=d)
    sigma_y = 0.03
    y = a @ x0 + sigma_y * jax.random.normal(key_n, (32,))

    x_init = 20.0 * jax.random.normal(jax.random.PRNGKey(99), (d,))
    residual_before = float(jnp.linalg.norm(a @ x_init - y))
    mse_before = float(jnp.mean((x_init - x0) ** 2))

    sigmas = edm_sigmas(24, sigma_max=20.0, sigma_min=0.1, rho=7.0)
    xhat = dcts(
        key_s,
        gaussian_ve_denoiser(prior_var=1.0),
        y,
        a,
        sigmas,
        sigma_y=sigma_y,
        m_steps=3,
        sigma_src=1.0,
        kappa=1.0,
    )
    residual_after = float(jnp.linalg.norm(a @ xhat - y))
    mse_after = float(jnp.mean((xhat - x0) ** 2))

    print("DCTS smoke (Gaussian A, d=32, well-determined)")
    print(f"residual ||A x - y|| before: {residual_before:.6f}")
    print(f"residual ||A xhat - y|| after: {residual_after:.6f}")
    print(f"reconstruction MSE before: {mse_before:.6f}")
    print(f"reconstruction MSE after:  {mse_after:.6f}")


if __name__ == "__main__":
    main()
