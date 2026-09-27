"""Stage I transmittance extraction, eqs (16)-(18) and (28)-(32)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from teda.operators import Stage1Ops

# Floor so the gate stays defined when sigma = 0 (noise-free tests).
_TAU_G_FLOOR = 1e-6
_ABS_Y_FLOOR = 1e-30


class ExtractResult(NamedTuple):
    tau_hat: jnp.ndarray
    b_hat: jnp.ndarray
    g: jnp.ndarray
    P_g: jnp.ndarray
    delta: jnp.ndarray
    z: jnp.ndarray
    tau_g: jnp.ndarray


def log_signal(Y):
    """z_k = ln |Y_k|, eq (17)."""
    Y = jnp.asarray(Y, dtype=jnp.float64)
    return jnp.log(jnp.maximum(jnp.abs(Y), _ABS_Y_FLOOR))


def noise_scale_tau_g(Y, sigma, c=1.0):
    """First-order noise-derived gate scale.

    Paper: tau_g is a first-order noise-derived scale. Supplementary
    quantization, sign guard, and dead zone are not implemented.

    Choice used here: tau_g = c * sigma / median(|Y|), the log-domain
    noise scale dz ~ sigma / |Y|, with a small positive floor.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    med = jnp.median(jnp.abs(Y))
    return jnp.maximum(c * sigma / (med + _ABS_Y_FLOOR), _TAU_G_FLOOR)


def gate(delta, tau_g):
    """g = 1 / (1 + |Delta| / tau_g)^8, eq (29)."""
    delta = jnp.asarray(delta, dtype=jnp.float64)
    tau_g = jnp.asarray(tau_g, dtype=jnp.float64)
    return 1.0 / (1.0 + jnp.abs(delta) / tau_g) ** 8


def absorption_disagreement(alpha, b_high, b_low):
    """Delta = alpha^T (b_high - b_low) / (alpha^T alpha), eq (28)."""
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    diff = jnp.asarray(b_high, dtype=jnp.float64) - jnp.asarray(b_low, dtype=jnp.float64)
    return (alpha @ diff) / (alpha @ alpha)


def extract_transmittance(z, ops: Stage1Ops, tau_g) -> ExtractResult:
    """Two-branch gated baseline and transmittance, eqs (30)-(32).

    b_hat = (1-g) H_low z + g H_high z
    P_g = (1-g) P_low + g P_high
    tau_hat = exp(P_g z)
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    tau_g = jnp.asarray(tau_g, dtype=jnp.float64)
    b_low = ops.H_low @ z
    b_high = ops.H_high @ z
    delta = absorption_disagreement(ops.alpha, b_high, b_low)
    g = gate(delta, tau_g)
    b_hat = (1.0 - g) * b_low + g * b_high
    P_g = (1.0 - g) * ops.P_low + g * ops.P_high
    tau_hat = jnp.exp(P_g @ z)
    return ExtractResult(
        tau_hat=tau_hat,
        b_hat=b_hat,
        g=g,
        P_g=P_g,
        delta=delta,
        z=z,
        tau_g=tau_g,
    )


def extract_from_radiance(y, B_air, ops: Stage1Ops, sigma, c_tau_g=1.0) -> ExtractResult:
    """Y = y - B(T_air), z = ln |Y|, then extract_transmittance, eqs (16)-(17)."""
    y = jnp.asarray(y, dtype=jnp.float64)
    B_air = jnp.asarray(B_air, dtype=jnp.float64)
    Y = y - B_air
    z = log_signal(Y)
    tau_g = noise_scale_tau_g(Y, sigma, c=c_tau_g)
    return extract_transmittance(z, ops, tau_g)
