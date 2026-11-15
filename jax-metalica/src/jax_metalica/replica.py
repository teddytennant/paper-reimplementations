"""Replica-exchange kernel over diffusion levels (eq 7 / 8 / 14, Appendix B)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_metalica.gmm import ToyGMM
from jax_metalica.kernels import (
    forward_mean,
    forward_var,
    log_normal,
    log_pi,
    pretrained_backward_mean,
    tilted_backward_mean,
)
from jax_metalica.sde import beta

Array = jax.Array


def swap_log_ratio(
    x_lo: Array,
    x_hi: Array,
    y_lo: Array,
    y_hi: Array,
    t_lo: Array | float,
    t_hi: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """log of the MH ratio in eq 8 / 14 (unclipped)."""
    dt = t_hi - t_lo
    v_fwd = forward_var(t_lo, dt)
    v_bwd = beta(t_hi) * dt

    log_pi_num = log_pi(y_lo, t_lo, omega, V, gmm, rho) + log_pi(
        y_hi, t_hi, omega, V, gmm, rho
    )
    log_pi_den = log_pi(x_lo, t_lo, omega, V, gmm, rho) + log_pi(
        x_hi, t_hi, omega, V, gmm, rho
    )

    log_F_num = log_normal(x_hi, forward_mean(y_lo, t_lo, dt), v_fwd)
    log_F_den = log_normal(y_hi, forward_mean(x_lo, t_lo, dt), v_fwd)

    mu_q_y = tilted_backward_mean(y_hi, t_hi, dt, omega, V, gmm, rho)
    mu_q_x = tilted_backward_mean(x_hi, t_hi, dt, omega, V, gmm, rho)
    log_B_num = log_normal(x_lo, mu_q_y, v_bwd)
    log_B_den = log_normal(y_lo, mu_q_x, v_bwd)

    return (log_pi_num - log_pi_den) + (log_B_num + log_F_num) - (log_B_den + log_F_den)


def swap_acceptance(
    x_lo: Array,
    x_hi: Array,
    y_lo: Array,
    y_hi: Array,
    t_lo: Array | float,
    t_hi: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> Array:
    """A = 1 ∧ exp(log ratio), clipped to [0, 1]."""
    log_A = swap_log_ratio(
        x_lo, x_hi, y_lo, y_hi, t_lo, t_hi, omega, V, gmm, rho
    )
    return jnp.clip(jnp.exp(jnp.clip(log_A, -80.0, 80.0)), 0.0, 1.0)


def attempt_swap(
    key: Array,
    x_lo: Array,
    x_hi: Array,
    t_lo: Array | float,
    t_hi: Array | float,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float = 0.5,
) -> tuple[Array, Array, Array]:
    """One adjacent EM proposal + Metropolis–Hastings swap.

    Returns ``(new_lo, new_hi, A)``.
    """
    dt = t_hi - t_lo
    k_f, k_b, k_a = jax.random.split(key, 3)
    y_hi = forward_mean(x_lo, t_lo, dt) + jnp.sqrt(
        jnp.maximum(forward_var(t_lo, dt), 1e-12)
    ) * jax.random.normal(k_f)
    mu_q = tilted_backward_mean(x_hi, t_hi, dt, omega, V, gmm, rho)
    v_bwd = jnp.maximum(beta(t_hi) * dt, 1e-12)
    y_lo = mu_q + jnp.sqrt(v_bwd) * jax.random.normal(k_b)
    A = swap_acceptance(x_lo, x_hi, y_lo, y_hi, t_lo, t_hi, omega, V, gmm, rho)
    u = jax.random.uniform(k_a)
    accept = u < A
    new_lo = jnp.where(accept, y_lo, x_lo)
    new_hi = jnp.where(accept, y_hi, x_hi)
    return new_lo, new_hi, A


def init_chain(key: Array, times: Array, gmm: ToyGMM) -> Array:
    """Reverse pass from p_1 = N(0,1) with pretrained backward kernels (Alg. 1)."""
    key, k_top = jax.random.split(key)
    x_top = jax.random.normal(k_top)
    dts = times[1:] - times[:-1]
    t_his = times[1:]
    n_steps = times.shape[0] - 1
    keys = jax.random.split(key, n_steps)

    def body(x: Array, elems: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        k_i, t_hi, dt = elems
        mu = pretrained_backward_mean(x, t_hi, dt, gmm)
        v = jnp.maximum(beta(t_hi) * dt, 1e-12)
        x_new = mu + jnp.sqrt(v) * jax.random.normal(k_i)
        return x_new, x_new

    _, xs = jax.lax.scan(body, x_top, (keys, t_his[::-1], dts[::-1]))
    # xs = [x_{M-1}, …, x_0]; reverse and append the top replica.
    return jnp.concatenate([xs[::-1], x_top[None]])


def replica_step(
    key: Array,
    chain: Array,
    times: Array,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float,
    parity: Array | int,
) -> Array:
    """One CREPE-style even/odd adjacent sweep, then redraw X^M ~ p_1.

    ``parity`` selects pairs (m-1, m) with (m-1) ≡ parity (mod 2). Paper
    Appendix B uses m ≡ n (mod 2); both checkerboards mix adjacent levels.
    """
    n = chain.shape[0]
    key, k_top = jax.random.split(key)
    chain = chain.at[-1].set(jax.random.normal(k_top, dtype=chain.dtype))
    parity = jnp.asarray(parity)

    def one_index(ch: Array, i: int) -> Array:
        k_i = jax.random.fold_in(key, i)

        def do_swap(c: Array) -> Array:
            y_lo, y_hi, _A = attempt_swap(
                k_i, c[i], c[i + 1], times[i], times[i + 1], omega, V, gmm, rho
            )
            c = c.at[i].set(y_lo)
            return c.at[i + 1].set(y_hi)

        return jax.lax.cond((jnp.asarray(i) % 2) == parity, do_swap, lambda c: c, ch)

    for i in range(n - 1):
        chain = one_index(chain, i)
    return chain
