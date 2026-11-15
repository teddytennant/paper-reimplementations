"""Denoiser-compatible sufficient statistic and transition velocity.

Paper (VE, alpha_t ≡ 1), inner coordinate s in [0, 1):

    bar_sigma_s = (1 - s) sigma_src + s sigma_t'
    v_s = (sigma_t^{-2} + s^2 / bar_sigma_s^2)^{-1}
    x_tilde_s = v_s (x_t / sigma_t^2 + s r_s / bar_sigma_s^2)
    sigma_tilde_s = sqrt(v_s)
    c_s = kappa v_s

    b_s = (sigma_t' - sigma_src) / bar_sigma_s
    u_s = b_s r_s + (1 - s b_s) m_s^y
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def bar_sigma(s: Array, sigma_src: Array, sigma_next: Array) -> Array:
    """Noise scale of the linear inner path R_s = s Z + bar_sigma_s eta."""
    return (1.0 - s) * sigma_src + s * sigma_next


def precision_variance(s: Array, sigma_t: Array, sigma_bar_s: Array) -> Array:
    """v_s in Eq. 7: posterior variance of Z given (X_t, R_s) under a flat prior."""
    prec_t = 1.0 / (sigma_t**2)
    prec_r = (s**2) / (sigma_bar_s**2)
    return 1.0 / (prec_t + prec_r)


def sufficient_statistic(
    x_t: Array,
    r_s: Array,
    s: Array,
    sigma_t: Array,
    sigma_src: Array,
    sigma_next: Array,
) -> tuple[Array, Array, Array]:
    """Eq. 7: (x_tilde_s, sigma_tilde_s, v_s).

    At s = 0 this reduces to (x_t, sigma_t, sigma_t^2).
    """
    sb = bar_sigma(s, sigma_src, sigma_next)
    v = precision_variance(s, sigma_t, sb)
    x_tilde = v * (x_t / (sigma_t**2) + s * r_s / (sb**2))
    sigma_tilde = jnp.sqrt(v)
    return x_tilde, sigma_tilde, v


def local_covariance(v_s: Array, kappa: Array) -> Array:
    """c_s = kappa v_s, isotropic local covariance scale."""
    return kappa * v_s


def flow_coefficient(s: Array, sigma_src: Array, sigma_next: Array) -> Array:
    """b_s = (sigma_t' - sigma_src) / bar_sigma_s for s in [0, 1)."""
    sb = bar_sigma(s, sigma_src, sigma_next)
    return (sigma_next - sigma_src) / sb


def flow_velocity(
    r_s: Array,
    m_y: Array,
    s: Array,
    sigma_src: Array,
    sigma_next: Array,
) -> Array:
    """Eqs. 12–13: u_s^y = b_s r_s + (1 - s b_s) m_s^y."""
    b = flow_coefficient(s, sigma_src, sigma_next)
    return b * r_s + (1.0 - s * b) * m_y
