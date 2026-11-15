"""Stage II operator-consistent distance alignment, eqs (34)-(36)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from teda.extract import ExtractResult, extract_from_radiance
from teda.operators import Stage1Ops


class MatchResult(NamedTuple):
    d_hat: jnp.ndarray
    cost: jnp.ndarray
    weights: jnp.ndarray
    tau_tilde: jnp.ndarray


class TEDAResult(NamedTuple):
    d_hat: jnp.ndarray
    tau_hat: jnp.ndarray
    b_hat: jnp.ndarray
    g: jnp.ndarray
    weights: jnp.ndarray
    cost: jnp.ndarray


def contrast_weights(b_hat):
    """w_k = exp(2 b_hat_k) / sum_j exp(2 b_hat_j), eq (35)."""
    b_hat = jnp.asarray(b_hat, dtype=jnp.float64)
    e = jnp.exp(2.0 * b_hat)
    return e / jnp.sum(e)


def operator_consistent_transmittance(ln_tau, P_g):
    """tau_tilde(d; g) = exp(P_g ln tau(d)), eq (34).

    ln_tau is (N,) for one distance or (n_d, N) for a bank.
    """
    ln_tau = jnp.asarray(ln_tau, dtype=jnp.float64)
    P_g = jnp.asarray(P_g, dtype=jnp.float64)
    if ln_tau.ndim == 1:
        return jnp.exp(P_g @ ln_tau)
    return jnp.exp(ln_tau @ P_g.T)


def weighted_mse(tau_hat, tau_model, weights):
    """sum_k w_k (tau_hat_k - tau_model_k)^2, eq (36) inner term."""
    residual = tau_hat - tau_model
    return jnp.sum(weights * residual**2, axis=-1)


def match_distance(tau_hat, tau_tilde_bank, weights, d_grid) -> MatchResult:
    """d_hat = argmin_d sum_k w_k (tau_hat_k - tau_tilde_k(d))^2, eq (36)."""
    tau_hat = jnp.asarray(tau_hat, dtype=jnp.float64)
    tau_tilde_bank = jnp.asarray(tau_tilde_bank, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    d_grid = jnp.asarray(d_grid, dtype=jnp.float64)
    cost = weighted_mse(tau_hat, tau_tilde_bank, weights)
    idx = jnp.argmin(cost)
    return MatchResult(
        d_hat=d_grid[idx],
        cost=cost,
        weights=weights,
        tau_tilde=tau_tilde_bank,
    )


def range_from_extract(
    extracted: ExtractResult,
    tau_bank,
    d_grid,
) -> MatchResult:
    """Apply P_g to the model bank, then weighted grid search."""
    ln_tau_bank = jnp.log(jnp.maximum(jnp.asarray(tau_bank, dtype=jnp.float64), 1e-30))
    tau_tilde = operator_consistent_transmittance(ln_tau_bank, extracted.P_g)
    w = contrast_weights(extracted.b_hat)
    return match_distance(extracted.tau_hat, tau_tilde, w, d_grid)


def run_pixel(y, B_air, ops: Stage1Ops, tau_bank, d_grid, sigma, c_tau_g=1.0) -> TEDAResult:
    """Full TEDA on one spectrum: extract then align."""
    extracted = extract_from_radiance(y, B_air, ops, sigma, c_tau_g=c_tau_g)
    matched = range_from_extract(extracted, tau_bank, d_grid)
    return TEDAResult(
        d_hat=matched.d_hat,
        tau_hat=extracted.tau_hat,
        b_hat=extracted.b_hat,
        g=extracted.g,
        weights=matched.weights,
        cost=matched.cost,
    )


def run_batch(y_batch, B_air, ops: Stage1Ops, tau_bank, d_grid, sigma, c_tau_g=1.0) -> TEDAResult:
    """Vectorized TEDA over pixels via jax.vmap (no Python loop over pixels)."""

    def _one(y):
        return run_pixel(y, B_air, ops, tau_bank, d_grid, sigma, c_tau_g=c_tau_g)

    return jax.vmap(_one)(jnp.asarray(y_batch, dtype=jnp.float64))
