"""Synthetic 1D trajectory: recover a smooth signal from quantized observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from quake.calibrate import observation_params, process_params, process_residual
from quake.sampler import correct_loop
from quake.transition import lagrange_transition, shift_register


def main() -> None:
    T = 24
    window = 2
    key = jax.random.PRNGKey(0)

    t = jnp.arange(T, dtype=jnp.float32)
    f = jnp.sin(t / 3.0) + 0.15 * t / T
    gamma_true, xi_true, R_true = 0.85, 0.12, 0.04
    noise = jax.random.normal(key, (T,)) * jnp.sqrt(R_true)
    f_hat = gamma_true * f + xi_true + noise

    gamma, xi, R = observation_params(f, f_hat, axis=0)
    lambdas = -t  # dummy decreasing log-SNR
    A, b, G = shift_register(jnp.array([1.2, -0.2], dtype=jnp.float32), 0.0, window)
    # Residuals of a lag-2 linear predictor on the true trajectory.
    history = jnp.stack([f[:-1], jnp.concatenate([f[:1], f[:-2]])], axis=-1)[1:]
    w = process_residual(f[2:], history, jnp.array([1.2, -0.2]), 0.0)
    mu, Q = process_params(w, axis=0)

    init_mean = jnp.array([f[0], f[0]], dtype=jnp.float32)
    init_cov = jnp.eye(window, dtype=jnp.float32) * 0.5
    # Shared transition; observation stats from calibration.
    f_corr, _state = correct_loop(
        f_hat,
        init_mean,
        init_cov,
        A,
        b,
        G,
        mu,
        jnp.maximum(Q, 1e-4),
        gamma,
        xi,
        jnp.maximum(R, 1e-6),
    )
    mse_raw = float(jnp.mean((f_hat - f) ** 2))
    mse_corr = float(jnp.mean((f_corr - f) ** 2))
    print("QuAKE synthetic 1D demo")
    print(f"  calibrated γ={float(gamma):.3f} ξ={float(xi):.3f} R={float(R):.4f}")
    print(f"  MSE quantized={mse_raw:.4f}  corrected={mse_corr:.4f}")
    _ = lagrange_transition(jnp.array([0.0, -1.0, -2.0], dtype=jnp.float32))


if __name__ == "__main__":
    main()
