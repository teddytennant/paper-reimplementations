"""Closed-form discrete coefficients k_t for the three source families."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np

# Physicists' Gauss-Hermite nodes for weight exp(-x^2).
_GH_N = 96
_GH_X_NP, _GH_W_NP = np.polynomial.hermite.hermgauss(_GH_N)


def continuous_ratio(k_tilde):
    """ct = k_tilde / (1 - k_tilde), with ct = +inf when k_tilde = 1."""
    k = jnp.asarray(k_tilde)
    one = k >= jnp.array(1.0, dtype=k.dtype)
    dummy = jnp.array(0.5, dtype=k.dtype)
    k_safe = jnp.where(one, dummy, k)
    inf = jnp.array(jnp.inf, dtype=k.dtype)
    return jnp.where(one, inf, k_safe / (1.0 - k_safe))


def _require_vocab(vocab_size):
    v = int(vocab_size)
    if v < 2:
        raise ValueError("vocab_size must be >= 2 (k_t is undefined for V=1)")
    return v


def k_gaussian(k_tilde, vocab_size, n_quad=None):
    """k_t^Gauss from eq 32-33 via Gauss-Hermite quadrature of Phi.

    rho = E_{G ~ N(0,1)} [ Phi(G + ct)^{V-1} ]
    k = (V * rho - 1) / (V - 1)
    """
    v = _require_vocab(vocab_size)
    k_tilde = jnp.asarray(k_tilde)
    orig = k_tilde.shape
    k_flat = k_tilde.reshape(-1)
    one = k_flat >= jnp.array(1.0, dtype=k_flat.dtype) - jnp.array(
        1e-6, dtype=k_flat.dtype
    )
    ct = continuous_ratio(k_flat)

    n_quad = int(n_quad) if n_quad is not None else _GH_N
    if n_quad == _GH_N:
        x_np, w_np = _GH_X_NP, _GH_W_NP
    else:
        x_np, w_np = np.polynomial.hermite.hermgauss(n_quad)
    x = jnp.asarray(x_np, dtype=k_flat.dtype)
    w = jnp.asarray(w_np, dtype=k_flat.dtype)
    g = jnp.sqrt(jnp.array(2.0, dtype=k_flat.dtype)) * x
    phi = norm.cdf(g[:, None] + ct[None, :])
    powered = jnp.where(one[None, :], jnp.array(1.0, dtype=k_flat.dtype), phi ** (v - 1))
    rho = jnp.sum(w[:, None] * powered, axis=0) / jnp.sqrt(
        jnp.array(np.pi, dtype=k_flat.dtype)
    )
    kt = (v * rho - 1.0) / (v - 1.0)
    kt = jnp.where(one, jnp.array(1.0, dtype=k_flat.dtype), kt)
    return jnp.clip(kt, 0.0, 1.0).reshape(orig)


def k_uniform(k_tilde, vocab_size, eps=1.0):
    """k_t^Unif for bounded uniform of width eps (eq 34).

    r_t = clip(ct / eps, 0, 1)
    k = (V r_t - r_t^V) / (V - 1)
    Independent of the interval location.
    """
    v = _require_vocab(vocab_size)
    eps = jnp.asarray(eps)
    k_tilde = jnp.asarray(k_tilde)
    orig = k_tilde.shape
    k_flat = k_tilde.reshape(-1)
    one = k_flat >= jnp.array(1.0, dtype=k_flat.dtype) - jnp.array(
        1e-6, dtype=k_flat.dtype
    )
    ct = continuous_ratio(k_flat)
    r = jnp.clip(ct / eps, 0.0, 1.0)
    r = jnp.where(one, jnp.array(1.0, dtype=k_flat.dtype), r)
    kt = (v * r - r**v) / (v - 1)
    return kt.reshape(orig)


def k_negexp(k_tilde, beta=1.0, vocab_size=None):
    """k_t^NegExp = 1 - exp(-beta * ct) (eq 35 / 61). Independent of V."""
    del vocab_size
    beta = jnp.asarray(beta)
    k_tilde = jnp.asarray(k_tilde)
    orig = k_tilde.shape
    k_flat = k_tilde.reshape(-1)
    one = k_flat >= jnp.array(1.0, dtype=k_flat.dtype) - jnp.array(
        1e-6, dtype=k_flat.dtype
    )
    ct = continuous_ratio(k_flat)
    ct_safe = jnp.where(one, jnp.array(0.0, dtype=k_flat.dtype), ct)
    kt = 1.0 - jnp.exp(-beta * ct_safe)
    kt = jnp.where(one, jnp.array(1.0, dtype=k_flat.dtype), kt)
    return kt.reshape(orig)


def k_t(k_tilde, vocab_size, family, *, eps=1.0, beta=1.0, n_quad=None):
    """Closed-form k_t for a named source family (linear schedule: k_tilde = t)."""
    family = str(family).lower()
    if family in ("gaussian", "gauss"):
        return k_gaussian(k_tilde, vocab_size, n_quad=n_quad)
    if family in ("uniform", "bounded_uniform", "unif"):
        return k_uniform(k_tilde, vocab_size, eps=eps)
    if family in ("negexp", "centered_negexp", "negative_exponential"):
        return k_negexp(k_tilde, beta=beta, vocab_size=vocab_size)
    raise ValueError(f"unknown source family: {family}")
