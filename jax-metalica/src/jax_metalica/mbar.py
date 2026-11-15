"""Multistate Bennett acceptance ratio reweighting (eq 17–18)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def mbar_update(f: Array, V_matrix: Array, n_counts: Array) -> Array:
    """One self-consistent MBAR step for f̂, pinning f̂_0 = 0.

    V_matrix[ℓ, i] = V_ℓ(ξ(x_i)), n_counts[ℓ] = N_ℓ.
    """
    log_n = jnp.log(n_counts)
    log_denom = jax.nn.logsumexp(log_n[:, None] + f[:, None] - V_matrix, axis=0)
    log_Z = jax.nn.logsumexp(-V_matrix - log_denom[None, :], axis=1)
    f_new = -log_Z
    return f_new - f_new[0]


def solve_mbar(
    V_matrix: Array,
    n_counts: Array,
    n_iter: int = 200,
) -> Array:
    """Iterate eq 17 until a fixed number of steps (tiny tests converge early)."""
    n_states = V_matrix.shape[0]
    f0 = jnp.zeros((n_states,), dtype=V_matrix.dtype)

    def body(_i: Array, f: Array) -> Array:
        return mbar_update(f, V_matrix, n_counts)

    return jax.lax.fori_loop(0, n_iter, body, f0)


def mbar_weights(V_matrix: Array, n_counts: Array, f: Array | None = None) -> Array:
    """Unnormalized-then-normalized MBAR weights (eq 18).

    w(x) ∝ 1 / ∑_ℓ N_ℓ exp(f̂_ℓ - V_ℓ(ξ(x))), ∑_i w_i = 1, w_i ≥ 0.
    """
    if f is None:
        f = solve_mbar(V_matrix, n_counts)
    log_n = jnp.log(n_counts)
    log_denom = jax.nn.logsumexp(log_n[:, None] + f[:, None] - V_matrix, axis=0)
    return jax.nn.softmax(-log_denom)


def free_energy_difference(
    samples: Array,
    weights: Array,
    eps: float = 1e-30,
) -> Array:
    """ΔF^ξ = log(weight in Ω_A / weight in Ω_B) with Ω_A={ω<0}, Ω_B={ω>0}.

    True value on the toy is η log 10. ``eps`` keeps the log finite if a
    basin is empty in a tiny run.
    """
    w_a = jnp.sum(jnp.where(samples < 0.0, weights, 0.0))
    w_b = jnp.sum(jnp.where(samples > 0.0, weights, 0.0))
    return jnp.log(w_a + eps) - jnp.log(w_b + eps)


def true_delta_F(eta: float | Array) -> Array:
    """Paper: true ΔF^ξ = η log 10."""
    return jnp.asarray(eta) * jnp.log(10.0)
