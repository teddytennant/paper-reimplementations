"""Seeded synthetic MercerFlow training on CPU. Small window, few steps."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax

from mercerflow import (
    euler_forecast,
    gram_matrix,
    init_velocity_mlp,
    mercer_basis,
    sample_prior,
    training_step,
)


def main():
    key = jax.random.PRNGKey(0)
    length = 16
    n_past = 8
    length_scale = 4.0
    jitter = 1e-3
    K = gram_matrix(length, length_scale, jitter)
    Phi, _ = mercer_basis(K)
    key, key_init, key_data = jax.random.split(key, 3)
    params = init_velocity_mlp(key_init, length, hidden=16)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    windows = sample_prior(key_data, K, n=16)
    y_past = windows[:, :n_past]
    y_future = windows[:, n_past:]
    loss = None
    for _ in range(5):
        key, key_step = jax.random.split(key)
        params, opt_state, loss = training_step(
            params,
            opt_state,
            optimizer,
            Phi,
            K,
            y_past,
            y_future,
            key_step,
        )
    key, key_forecast = jax.random.split(key)
    forecast = euler_forecast(
        params, Phi, y_past[0], K, key_forecast, n_steps=8
    )
    loss_value = float(loss)
    print(f"loss: {loss_value:.6f}")
    print(f"sample shape: {tuple(forecast.shape)}")
    if not jnp.isfinite(loss):
        raise SystemExit("loss is not finite")
    if tuple(forecast.shape) != (length - n_past,):
        raise SystemExit(f"unexpected sample shape {tuple(forecast.shape)}")


if __name__ == "__main__":
    main()
