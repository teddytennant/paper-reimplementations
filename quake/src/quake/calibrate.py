"""Timestep/channel calibration of observation and process statistics (eqs 77–80)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _mean(x, axis) -> Array:
    return jnp.mean(x, axis=axis)


def _var(x, axis) -> Array:
    m = jnp.mean(x, axis=axis, keepdims=True)
    return jnp.mean((x - m) ** 2, axis=axis)


def _cov(x, y, axis) -> Array:
    mx = jnp.mean(x, axis=axis, keepdims=True)
    my = jnp.mean(y, axis=axis, keepdims=True)
    return jnp.mean((x - mx) * (y - my), axis=axis)


def _expand_like(stat, samples, axis) -> Array:
    """Insert size-1 axes so reduced stats broadcast against `samples`."""
    axis_t = (axis,) if isinstance(axis, int) else tuple(axis)
    ndim = samples.ndim
    axis_t = tuple(sorted(a if a >= 0 else a + ndim for a in axis_t))
    expanded = jnp.asarray(stat)
    for a in axis_t:
        expanded = jnp.expand_dims(expanded, axis=a)
    return expanded


def observation_params(f, f_hat, axis=0, var_eps: float = 1e-12) -> tuple[Array, Array, Array]:
    """Linear-Gaussian observation parameters from paired tensors.

    `γ = Cov(f, f̂) / Var(f)`,  `ξ = E[f̂] − γ E[f]`,
    `R = Var(f̂ − γ f − ξ)`  (eqs 77–79 / 42).
    """
    f = jnp.asarray(f)
    f_hat = jnp.asarray(f_hat)
    var_f = _var(f, axis)
    cov = _cov(f, f_hat, axis)
    gamma = cov / jnp.maximum(var_f, jnp.asarray(var_eps, dtype=f.dtype))
    xi = _mean(f_hat, axis) - gamma * _mean(f, axis)
    residual = f_hat - _expand_like(gamma, f, axis) * f - _expand_like(xi, f, axis)
    R = _var(residual, axis)
    return gamma, xi, R


def process_residual(f_current, history, coeffs, bias=0.0) -> Array:
    """`w_i = f_i − (a · history + b)` with history `(f_{i-1}, ..., f_{i-H})`."""
    f_current = jnp.asarray(f_current)
    history = jnp.asarray(history)
    coeffs = jnp.asarray(coeffs)
    pred = jnp.sum(coeffs * history, axis=-1) + jnp.asarray(bias, dtype=f_current.dtype)
    return f_current - pred


def process_params(residual, axis=0) -> tuple[Array, Array]:
    """`μ = E[w]`, `Q = Var(w)` (eq 80)."""
    residual = jnp.asarray(residual)
    return _mean(residual, axis), _var(residual, axis)
