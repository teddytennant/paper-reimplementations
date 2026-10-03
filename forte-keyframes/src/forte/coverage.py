"""Logarithmic temporal coverage, Equation (3), and its marginal gain."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _as_times(timestamps):
    times = jnp.asarray(timestamps, dtype=jnp.float64).reshape(-1)
    return jnp.unique(times)


def coverage(timestamps, T, tau):
    """Coverage C_tau(S) = sum_i log(1 + g_i / tau).

    S is an ordered set in [0, T]. Boundaries are s_0 = 0 and s_{m+1} = T,
    and g_i = s_{i+1} - s_i. The sum uses log1p.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    horizon = float(T)
    if horizon < 0:
        raise ValueError("T must be nonnegative")
    times = _as_times(timestamps)
    bounds = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.float64),
            times,
            jnp.asarray([horizon], dtype=jnp.float64),
        ]
    )
    gaps = jnp.diff(bounds)
    return float(jnp.sum(jnp.log1p(gaps / tau)))


def marginal_gains(candidates, selected, T, tau):
    """Closed-form coverage gain of inserting each candidate into selected.

    If t splits a gap g into a and b with a + b = g, the gain is
    log1p(a / tau) + log1p(b / tau) - log1p(g / tau). Gains are nonnegative
    for t inside [0, T]. A timestamp already in the set has gain zero.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    horizon = jnp.asarray(T, dtype=jnp.float64)
    cand = jnp.asarray(candidates, dtype=jnp.float64).reshape(-1)
    selected_times = _as_times(selected)
    if selected_times.size == 0:
        left = jnp.zeros_like(cand)
        right = jnp.full_like(cand, horizon)
    else:
        idx = jnp.searchsorted(selected_times, cand, side="left")
        left_idx = jnp.clip(idx - 1, 0, selected_times.size - 1)
        right_idx = jnp.clip(idx, 0, selected_times.size - 1)
        left = jnp.where(idx == 0, 0.0, selected_times[left_idx])
        right = jnp.where(idx == selected_times.size, horizon, selected_times[right_idx])
    a = cand - left
    b = right - cand
    gap = right - left
    gain = jnp.log1p(a / tau) + jnp.log1p(b / tau) - jnp.log1p(gap / tau)
    valid = (a >= 0.0) & (b >= 0.0) & jnp.isfinite(gain)
    return jnp.where(valid, gain, 0.0)


def marginal_gain(t, selected, T, tau):
    """Coverage gain of adding one timestamp t to selected."""
    gains = marginal_gains(jnp.asarray([t], dtype=jnp.float64), selected, T, tau)
    return float(gains[0])
