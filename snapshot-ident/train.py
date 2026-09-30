"""CPU smoke: tiny planted-curl OU, print JSON relErr_weak and relErr_grad."""
from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from snapshot_ident._host import preload

preload()

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from snapshot_ident.estimator import gradient_only_fit, rel_err_solenoidal, ridge_fit
from snapshot_ident.moments import weak_design_from_second_moments
from snapshot_ident.sde import planted_curl_H, rotation_generator, snapshot_covariances


def main() -> None:
    d, m = 4, 2
    nu = 0.5
    sigma2 = 0.5
    omega = 0.8
    times = jnp.array([0.0, 0.5, 1.0, 1.5], dtype=jnp.float64)
    Sigma0 = jnp.diag(jnp.array([2.8, 0.22, 1.3, 0.7], dtype=jnp.float64))

    H_star = planted_curl_H(d, m, nu, omega)
    Omega = rotation_generator(d, m, omega)
    Sigmas = snapshot_covariances(Sigma0, times, nu, sigma2, Omega)
    dts = jnp.diff(times)
    X, bb = weak_design_from_second_moments(Sigmas, dts, sigma2)

    H_weak = ridge_fit(X, bb, d, zeta=1e-10, restrict="full")
    H_grad = gradient_only_fit(X, bb, d, zeta=1e-10)

    payload = {
        "relErr_weak": float(rel_err_solenoidal(H_weak, H_star)),
        "relErr_grad": float(rel_err_solenoidal(H_grad, H_star)),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
