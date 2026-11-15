"""Generate synthetic noisy-grid FLR data, reconstruct, fit, predict."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from flr_noisy import fit_flr, fourier_coefficients, generate_synthetic, predict


def main() -> None:
    key = jax.random.PRNGKey(0)
    n_train, n_test, p = 64, 32, 48
    recon_dim, max_m = 11, 5
    key_tr, key_te = jax.random.split(key)
    train = generate_synthetic(
        key_tr,
        n_train,
        p,
        n_kl=21,
        eigenvalue_decay=4.0,
        beta="fourier",
        beta_decay=2.0,
        beta_dim=7,
        sigma2=0.1,
        tau2=0.1,
    )
    test = generate_synthetic(
        key_te,
        n_test,
        p,
        n_kl=21,
        eigenvalue_decay=4.0,
        beta="fourier",
        beta_decay=2.0,
        beta_dim=7,
        sigma2=0.1,
        tau2=0.1,
    )
    fit = fit_flr(
        train.Z,
        train.y,
        recon_dim=recon_dim,
        max_m=max_m,
        sigma2=train.sigma2,
    )
    test_coeffs = fourier_coefficients(test.Z, recon_dim)
    pred = np.asarray(predict(fit.alpha, test_coeffs))
    mse = float(np.mean((pred - test.y) ** 2))
    naive = float(np.mean((train.y.mean() - test.y) ** 2))
    print(f"n_train={n_train} n_test={n_test} p={p} recon_dim={recon_dim}")
    print(f"m_hat={fit.m_hat} dim={fit.dim} on_gbar={fit.on_gbar} plugin={fit.plugin}")
    print(f"prediction MSE={mse:.6f}")
    print(f"naive mean MSE={naive:.6f}")
    print(f"criteria={np.array2string(fit.criteria, precision=4)}")


if __name__ == "__main__":
    main()
