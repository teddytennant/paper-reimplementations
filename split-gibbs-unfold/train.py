"""CPU/GPU smoke: train the unfolded likelihood denoiser, write results.json."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from sgs_unfold.likelihood import make_likelihood_denoiser, train_unfolded, unfolded_denoise
from sgs_unfold.phase import make_A, measurement_residual, observe
from sgs_unfold.prior import make_prior_denoiser
from sgs_unfold.sampler import decreasing_t_grid, split_gibbs_sample, tau_schedule
from sgs_unfold.schedule import edm_sigma, edm_sigma_dot_over_sigma


def main():
    n, m = 8, 16
    tau_0 = 1.0
    meas_std = 0.2
    key = jax.random.PRNGKey(0)
    kA, k_s, k_y, k_tr, k_sm = jax.random.split(key, 5)
    A = make_A(kA, m, n)
    s_true = tau_0 * jax.random.normal(k_s, (n,))
    y = observe(k_y, A, s_true, meas_std)

    params, losses = train_unfolded(
        k_tr,
        A,
        tau_0,
        meas_std,
        steps=25,
        batch=32,
        lr=1e-2,
        n_layers=2,
        hidden=8,
    )
    loss_start = float(losses[0])
    loss_end = float(losses[-1])

    eta = 0.4
    x_noisy = s_true + eta * jax.random.normal(jax.random.fold_in(k_s, 1), (n,))
    s_hat = unfolded_denoise(params, x_noisy, eta, y, A, tau_0, meas_std)
    r_init = float(measurement_residual(x_noisy, y, A))
    r_denoise = float(measurement_residual(s_hat, y, A))
    mse_noisy = float(jnp.mean((x_noisy - s_true) ** 2))
    mse_denoise = float(jnp.mean((s_hat - s_true) ** 2))

    D_P = make_prior_denoiser(tau_0**2)
    D_L = make_likelihood_denoiser(params, y, A, tau_0, meas_std)
    t_grid = decreasing_t_grid(t_max=1.0, t_min=0.15, n_steps=4)
    taus = tau_schedule(0.45, 0.2, 3)
    s_samp, s_prime = split_gibbs_sample(
        k_sm,
        x_noisy,
        tau_0,
        taus,
        D_P,
        D_L,
        t_grid,
        edm_sigma,
        edm_sigma_dot_over_sigma,
    )
    r_samp = float(measurement_residual(s_samp, y, A))

    payload = {
        "n": n,
        "m": m,
        "unfold_layers": 2,
        "train_steps": 25,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "residual_noisy": r_init,
        "residual_after_likelihood_denoise": r_denoise,
        "mse_noisy": mse_noisy,
        "mse_denoise": mse_denoise,
        "residual_after_sampler": r_samp,
        "sampler_outer": 3,
        "s_finite": bool(jnp.all(jnp.isfinite(s_samp))),
        "s_prime_finite": bool(jnp.all(jnp.isfinite(s_prime))),
    }
    out = Path("results.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"sgs-unfold n={n}: loss {loss_start:.4f}->{loss_end:.4f} "
        f"mse {mse_noisy:.4f}->{mse_denoise:.4f} wrote {out}"
    )


if __name__ == "__main__":
    main()
