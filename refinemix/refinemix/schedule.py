"""Log-linear (MDLM) noise schedule from Appendix B.

Experimental schedule (eps = 1e-3):
    alpha(t) = 1 - (1 - eps) * t
    alpha(0) = 1, alpha(1) = eps, alpha'(t) = -(1 - eps).

The continuous-time NELBO weight alpha'(t) / (1 - alpha(t)) equals -1/t.

Theorem 1 starts from pure noise, which needs alpha(1) = 0. That idealized
schedule is `alpha(..., ideal=True)` / `alpha_ideal`, not the eps-truncated
schedule used in training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS = 1e-3


def alpha(t, eps: float = EPS, ideal: bool = False):
    """Keep-probability alpha(t). `ideal=True` sets alpha(1) = 0."""
    t = jnp.asarray(t)
    one = jnp.asarray(1.0, dtype=t.dtype)
    if ideal:
        return one - t
    eps_t = jnp.asarray(eps, dtype=t.dtype)
    return one - (one - eps_t) * t


def alpha_ideal(t):
    """Idealized kernel with alpha(1) = 0, used only for Theorem 1."""
    return alpha(t, ideal=True)


def alpha_prime(t, eps: float = EPS, ideal: bool = False):
    """Constant derivative of alpha. Shape matches `t`."""
    t = jnp.asarray(t)
    one = jnp.asarray(1.0, dtype=t.dtype)
    if ideal:
        value = -one
    else:
        value = -(one - jnp.asarray(eps, dtype=t.dtype))
    return jnp.broadcast_to(value, t.shape)


def nelbo_weight(t, eps: float = EPS, ideal: bool = False):
    """alpha'(t) / (1 - alpha(t)). Negative.

    On the log-linear schedule the (1 - eps) factors cancel, so the weight is
    exactly -1/t. The idealized schedule alpha(t) = 1 - t has the same closed
    form. Returning -1/t avoids a float32 cancellation in the ratio near t = 0.
    """
    del eps, ideal
    t = jnp.asarray(t)
    return -jnp.ones_like(t) / t


def sample_time(key, shape=(), mode: str = "experimental", eps: float = EPS):
    """Draw diffusion times.

    `experimental` (Appendix B, training default): t ~ Uniform(eps, 1).
    `algorithm` (Algorithm 1 as written): t ~ Uniform(0, 1).
    """
    mode = mode.lower()
    if mode in ("experimental", "uniform_eps", "u_eps"):
        return jax.random.uniform(key, shape, minval=eps, maxval=1.0)
    if mode in ("algorithm", "uniform", "u01", "u(0,1)"):
        return jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
    raise ValueError(
        f"unknown time sampler {mode!r}; use 'experimental' or 'algorithm'"
    )
