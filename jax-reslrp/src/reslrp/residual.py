"""LRP-0 and ResLRP LRP-γ on residual additions (Eq. 1, Eq. 3, Eq. 4)."""

from __future__ import annotations

import jax.numpy as jnp


def _same_sign(z, z_out):
    """Indicator δ_{sign(z)=sign(z_out)} via z * z_out > 0 (Sec. A.3)."""
    return (z * z_out > 0).astype(z.dtype)


def _gamma_weights(z_in, z_up, gamma):
    z_out = z_in + z_up
    gamma = jnp.asarray(gamma, dtype=z_in.dtype)
    w_in = 1.0 + gamma * _same_sign(z_in, z_out)
    w_up = 1.0 + gamma * _same_sign(z_up, z_out)
    return w_in, w_up, z_out


def phi(z_in, z_up, gamma, eps: float = 1e-12):
    """Residual amplification factor Φ(a, b) (Eq. 4).

    Φ is the factor scaling R_out onto the in-branch in Eq. 3. For
    z_out ≠ 0 and γ > 0 the paper proves −1/γ ≤ Φ ≤ 1 + 1/γ.
    When the γ-weighted denominator is degenerate, Φ = 1/2.
    """
    z_in = jnp.asarray(z_in)
    z_up = jnp.asarray(z_up)
    w_in, w_up, _ = _gamma_weights(z_in, z_up, gamma)
    numer = w_in * z_in
    denom = numer + w_up * z_up
    unstable = jnp.abs(denom) <= jnp.asarray(eps, dtype=z_in.dtype)
    return jnp.where(unstable, jnp.asarray(0.5, dtype=z_in.dtype), numer / denom)


def lrp0(z_in, z_up, r_out, eps: float = 0.0):
    """Standard LRP-ε/0 on a residual addition (Eq. 1).

    R_in = z_in / (z_out + sign(z_out) ε) * R_out, and likewise for R_up.
    With ε = 0 this is LRP-0 and explodes under cancellation.
    """
    z_in = jnp.asarray(z_in)
    z_up = jnp.asarray(z_up)
    r_out = jnp.asarray(r_out)
    z_out = z_in + z_up
    sign = jnp.sign(z_out)
    denom = z_out + jnp.asarray(eps, dtype=z_in.dtype) * sign
    r_in = z_in / denom * r_out
    r_up = z_up / denom * r_out
    return r_in, r_up


def lrp_gamma(z_in, z_up, r_out, gamma, eps: float = 1e-9):
    """ResLRP LRP-γ on a residual addition (Eq. 3).

    Favors branches whose contribution has the same sign as z_out.
    ``eps`` is a degenerate-denominator guard (Eq. 3 has no ε): if the
    γ-weighted sum is ~0, relevance is split equally. Otherwise
    R_in + R_up = R_out elementwise. γ = 0 recovers LRP-0 when z_out ≠ 0.
    """
    z_in = jnp.asarray(z_in)
    z_up = jnp.asarray(z_up)
    r_out = jnp.asarray(r_out)
    w_in, w_up, _ = _gamma_weights(z_in, z_up, gamma)
    num_in = w_in * z_in
    num_up = w_up * z_up
    denom = num_in + num_up
    unstable = jnp.abs(denom) <= jnp.asarray(eps, dtype=z_in.dtype)
    half = jnp.asarray(0.5, dtype=z_in.dtype)
    phi_in = jnp.where(unstable, half, num_in / denom)
    phi_up = jnp.where(unstable, half, num_up / denom)
    r_in = phi_in * r_out
    r_up = phi_up * r_out
    return r_in, r_up
