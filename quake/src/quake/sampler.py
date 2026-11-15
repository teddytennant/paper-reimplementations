"""Plug-in corrector: quantized `f̂_i` in, posterior mean of `f_i` out."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from quake.kalman import step


class FilterState(NamedTuple):
    mean: Array
    cov: Array


def _broadcasts(shape, sample_shape) -> bool:
    try:
        jnp.broadcast_shapes(shape, sample_shape)
    except ValueError:
        return False
    return True


def _stack_time(param, T: int, static_shape: tuple[int, ...]) -> Array:
    """Shared `static_shape` or already stacked `(T,) + static_shape`."""
    p = jnp.asarray(param)
    timed = (T,) + static_shape
    if p.shape == timed:
        return p
    return jnp.broadcast_to(p, timed)


def _stack_stat(param, T: int, sample_shape: tuple[int, ...]) -> Array:
    """Scalar/channel stats: shared, or leading `T` if that broadcasts with samples."""
    p = jnp.asarray(param)
    if p.shape[:1] == (T,) and _broadcasts(p.shape[1:], sample_shape):
        return p
    return jnp.broadcast_to(p, (T,) + p.shape)


def correct_step(
    state: FilterState,
    f_hat,
    A,
    b,
    G,
    mu,
    Q,
    gamma,
    xi,
    R,
    *,
    do_predict: bool = True,
) -> tuple[FilterState, Array]:
    """One Kalman step. Returns updated state and posterior mean of the current slot."""
    mean, cov, f_corr = step(
        state.mean,
        state.cov,
        f_hat,
        A,
        b,
        G,
        mu,
        Q,
        gamma,
        xi,
        R,
        do_predict=do_predict,
    )
    return FilterState(mean, cov), f_corr


def correct_loop(
    f_hats,
    init_mean,
    init_cov,
    A,
    b,
    G,
    mu,
    Q,
    gamma,
    xi,
    R,
) -> tuple[Array, FilterState]:
    """Run the plug-in corrector over a time sequence of quantized outputs.

    `f_hats` has shape `(T, ...)` matching trailing batch/channel dims of the
    filter state. Per-step parameters may lead with `T`; otherwise they are
    shared across time. The first step skips the transition (Algorithm 1).
    """
    f_hats = jnp.asarray(f_hats)
    init_mean = jnp.asarray(init_mean)
    init_cov = jnp.asarray(init_cov)
    T = int(f_hats.shape[0])
    window = int(init_mean.shape[-1])
    sample_shape = f_hats.shape[1:]
    A_t = _stack_time(A, T, (window, window))
    b_t = _stack_time(b, T, (window,))
    G_t = _stack_time(G, T, (window,))
    mu_t = _stack_stat(mu, T, sample_shape)
    Q_t = _stack_stat(Q, T, sample_shape)
    gamma_t = _stack_stat(gamma, T, sample_shape)
    xi_t = _stack_stat(xi, T, sample_shape)
    R_t = _stack_stat(R, T, sample_shape)

    state = FilterState(jnp.asarray(init_mean), jnp.asarray(init_cov))
    outs = []
    for t in range(T):
        state, f_corr = correct_step(
            state,
            f_hats[t],
            A_t[t],
            b_t[t],
            G_t[t],
            mu_t[t],
            Q_t[t],
            gamma_t[t],
            xi_t[t],
            R_t[t],
            do_predict=t > 0,
        )
        outs.append(f_corr)
    return jnp.stack(outs, axis=0), state
