"""Straight-through gates, bounded rates, and variance links."""

import jax
import jax.numpy as jnp

# Posterior / prior rate link (eq. 20): output lies in (1e-3, 10).
RATE_LO = 1e-3
RATE_HI = 10.0
RATE_A = float(jnp.log(RATE_LO))
RATE_B = float(jnp.log(RATE_HI))

# Retention floor and proposal slack.
RHO_MIN = 1e-2
PROPOSAL_RHO = 1e-3
TAU_ST = 1.0

LOG_VAR_MIN = -7.0
LOG_VAR_MAX = 3.0


def rate_link(ell, a=RATE_A, b=RATE_B):
    """lambda = exp(a + (b - a) * sigmoid(ell)), a = log(1e-3), b = log(10).

    The image is the open interval (1e-3, 10). Float32 sigmoid saturates, which
    would round onto the endpoints; those saturated values are nudged to the
    nearest representable interior point. Interior logits match the formula.
    """
    lam = jnp.exp(a + (b - a) * jax.nn.sigmoid(ell))
    lo = jnp.nextafter(jnp.asarray(RATE_LO, dtype=lam.dtype), jnp.asarray(jnp.inf, dtype=lam.dtype))
    hi = jnp.nextafter(jnp.asarray(RATE_HI, dtype=lam.dtype), jnp.asarray(0.0, dtype=lam.dtype))
    lam = jnp.where(lam <= RATE_LO, lo, lam)
    lam = jnp.where(lam >= RATE_HI, hi, lam)
    return lam


def logit(p):
    """log(p / (1 - p)). p must lie in (0, 1)."""
    p = jnp.asarray(p)
    return jnp.log(p) - jnp.log1p(-p)


def straight_through_gate(logit_p, zeta, tau=TAU_ST):
    """Hard-forward logistic gate with a soft straight-through gradient (eq. 9).

    Forward value equals the hard threshold 1{logit(p) + zeta > 0}.
    dg / d logit(p) = g_soft * (1 - g_soft) / tau.
    zeta is standard logistic noise.
    """
    logit_p = jnp.asarray(logit_p)
    zeta = jnp.asarray(zeta)
    hard = (logit_p + zeta > 0).astype(logit_p.dtype)
    soft = jax.nn.sigmoid((logit_p + zeta) / tau)
    # Parenthesize so the forward residual is an exact zero and g equals g_hard.
    # (hard + soft) - soft is not exact in float32 when hard is 1.
    return hard + (soft - jax.lax.stop_gradient(soft))


def proposal_rate(lambda_k, rho=PROPOSAL_RHO):
    """lambda_bar = stop_gradient(max_k lambda_k) + rho.

    lambda_k: (..., K). Reduction is over the last axis.
    """
    return jax.lax.stop_gradient(jnp.max(lambda_k, axis=-1)) + rho


def retention_from_raw(raw, rho_min=RHO_MIN):
    """R = rho_min + (1 - rho_min) * sigmoid(raw), so R is in (rho_min, 1)."""
    return rho_min + (1.0 - rho_min) * jax.nn.sigmoid(raw)


def variance_from_log(log_var):
    """Positive diagonal variance, log clipped to [-7, 3]."""
    return jnp.exp(jnp.clip(log_var, LOG_VAR_MIN, LOG_VAR_MAX))
