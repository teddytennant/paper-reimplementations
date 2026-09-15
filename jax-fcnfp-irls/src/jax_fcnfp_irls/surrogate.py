"""Lasry-Lions surrogate, IRLS weights, and mu_max (paper eqs. 1-6, 14)."""

from __future__ import annotations

import jax.numpy as jnp

S_ANCHOR = 1e-2
WEIGHT_CLIP = (1e-12, 1e12)
R_FLOOR = 1e-12


def _as_f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def log_tail_alpha(lam, lam0, s: float = S_ANCHOR):
    """alpha = lam / (lam + s * lam0), anchored to the start (paper Sec. III-A)."""
    lam = _as_f64(lam)
    lam0 = _as_f64(lam0)
    return lam / (lam + s * lam0)


def transition_params(r, b, lam, mu, alpha):
    """Per-edge T, x1, S, x2, psi2 from eq. (2).

    T = lam * r + sqrt(2 * lam * b)
    x1 = T * (lam - mu) / lam
    S = T - mu * r
    x2 = (S + sqrt(S^2 - 4 * mu * alpha * b)) / 2
    psi2 = T^2 / (2 lam) - (x2 - T)^2 / (2 mu)
    """
    r = _as_f64(r)
    b = _as_f64(b)
    lam = _as_f64(lam)
    mu = _as_f64(mu)
    alpha = _as_f64(alpha)
    T = lam * r + jnp.sqrt(jnp.maximum(2.0 * lam * b, 0.0))
    x1 = T * (lam - mu) / lam
    S = T - mu * r
    disc = S * S - 4.0 * mu * alpha * b
    disc = jnp.maximum(disc, 0.0)
    x2 = 0.5 * (S + jnp.sqrt(disc))
    psi2 = T * T / (2.0 * lam) - (x2 - T) * (x2 - T) / (2.0 * mu)
    return T, x1, S, x2, psi2


def mu_max(r, b, lam, alpha):
    """Admissible mu cap (eq. 14). When r=0 this is lam / (2 alpha)."""
    r = _as_f64(r)
    b = _as_f64(b)
    lam = _as_f64(lam)
    alpha = _as_f64(alpha)
    T = lam * r + jnp.sqrt(jnp.maximum(2.0 * lam * b, 0.0))
    A = 2.0 * T * r + 4.0 * alpha * b
    disc = A * A - 4.0 * (r * r) * (T * T)
    disc = jnp.maximum(disc, 0.0)
    denom = A + jnp.sqrt(disc)
    mu_r = 2.0 * T * T / jnp.maximum(denom, 1e-30)
    mu_r0 = lam / (2.0 * alpha)
    return jnp.where(r <= 0.0, mu_r0, mu_r)


def admissible_mu(r, b, lam, alpha, theta: float = 0.5):
    """mu = min{theta * lam, min_e mu_max,e(lam)} with 0 < mu < lam."""
    r = _as_f64(r)
    b = _as_f64(b)
    lam = float(lam)
    alpha = _as_f64(alpha)
    caps = mu_max(r, b, lam, alpha)
    cap = float(jnp.min(jnp.reshape(caps, (-1,))))
    mu = min(theta * lam, cap)
    mu = min(mu, 0.999999 * lam)
    return max(mu, 1e-16)


def ll_quadratic(x, lam, mu):
    """Quadratic branch: x^2 / (2 (lam - mu)) for |x| <= x1."""
    x = _as_f64(x)
    lam = _as_f64(lam)
    mu = _as_f64(mu)
    return (x * x) / (2.0 * (lam - mu))


def ll_middle(x, T, lam, mu):
    """Middle branch: T^2/(2 lam) - (|x| - T)^2 / (2 mu) for x1 < |x| < x2."""
    x = _as_f64(x)
    T = _as_f64(T)
    lam = _as_f64(lam)
    mu = _as_f64(mu)
    ax = jnp.abs(x)
    return T * T / (2.0 * lam) - (ax - T) * (ax - T) / (2.0 * mu)


def ll_log_tail(x, r, b, x2, psi2, alpha):
    """Log-tail: r(|x|-x2) + alpha b log(|x|/x2) + psi2 for |x| >= x2."""
    x = _as_f64(x)
    r = _as_f64(r)
    b = _as_f64(b)
    x2 = _as_f64(x2)
    psi2 = _as_f64(psi2)
    alpha = _as_f64(alpha)
    ax = jnp.abs(x)
    ax_safe = jnp.maximum(ax, 1e-30)
    x2_safe = jnp.maximum(x2, 1e-30)
    return r * (ax - x2) + alpha * b * jnp.log(ax_safe / x2_safe) + psi2


def ll_value(x, r, b, lam, mu, alpha):
    """Per-edge Lasry-Lions surrogate LL^{lam,mu}(x) (eq. 2)."""
    x = _as_f64(x)
    r = _as_f64(r)
    b = _as_f64(b)
    T, x1, _S, x2, psi2 = transition_params(r, b, lam, mu, alpha)
    ax = jnp.abs(x)
    quad = ll_quadratic(ax, lam, mu)
    mid = ll_middle(ax, T, lam, mu)
    tail = ll_log_tail(ax, r, b, x2, psi2, alpha)
    return jnp.where(ax <= x1, quad, jnp.where(ax < x2, mid, tail))


def ll_derivative(x, r, b, lam, mu, alpha):
    """d/dx of the even per-edge surrogate (odd function).

    For x > 0:
      x / (lam - mu)                 if x <= x1
      (T - x) / mu                   if x1 < x < x2
      r + alpha b / x                if x >= x2
    At x1 both one-sided derivatives equal T / lam.
    """
    x = _as_f64(x)
    r = _as_f64(r)
    b = _as_f64(b)
    T, x1, _S, x2, _psi2 = transition_params(r, b, lam, mu, alpha)
    ax = jnp.abs(x)
    ax_safe = jnp.maximum(ax, 1e-30)
    d_quad = ax / (lam - mu)
    d_mid = (T - ax) / mu
    d_tail = r + alpha * b / ax_safe
    dpos = jnp.where(ax <= x1, d_quad, jnp.where(ax < x2, d_mid, d_tail))
    return jnp.sign(x) * dpos


def ll_second_derivative(x, r, b, lam, mu, alpha):
    """Second derivative of LL (even). Quadratic region: 1/(lam-mu). Middle: -1/mu."""
    x = _as_f64(x)
    r = _as_f64(r)
    b = _as_f64(b)
    _T, x1, _S, x2, _psi2 = transition_params(r, b, lam, mu, alpha)
    ax = jnp.abs(x)
    ax_safe = jnp.maximum(ax, 1e-30)
    d2_quad = 1.0 / (lam - mu)
    d2_mid = -1.0 / mu
    d2_tail = -alpha * b / (ax_safe * ax_safe)
    return jnp.where(ax <= x1, d2_quad, jnp.where(ax < x2, d2_mid, d2_tail))


def j_ll(f, r, b, lam, mu, alpha):
    """Smoothed objective J_LL = sum_e LL_e(f_e) (eq. 3)."""
    return jnp.sum(ll_value(f, r, b, lam, mu, alpha))


def j_true(f, r, b, tau: float = 0.0):
    """True fixed-charge objective (eq. 1): sum (r_e f_e + b_e 1{|f_e|>tau})."""
    f = _as_f64(f)
    r = _as_f64(r)
    b = _as_f64(b)
    ax = jnp.abs(f)
    return jnp.sum(r * ax + b * (ax > tau).astype(jnp.float64))


def irls_weights(f, r, b, lam, mu, alpha, clip: bool = True):
    """IRLS weights (eq. 6), optionally clipped to [1e-12, 1e12].

    w(f) =
      1/(lam-mu)                      if |f| <= x1  (also used at f=0)
      (T - |f|) / (mu |f|)            if x1 < |f| < x2
      r/|f| + alpha b / |f|^2         if |f| >= x2
    """
    f = _as_f64(f)
    r = _as_f64(r)
    b = _as_f64(b)
    T, x1, _S, x2, _psi2 = transition_params(r, b, lam, mu, alpha)
    ax = jnp.abs(f)
    ax_safe = jnp.maximum(ax, 1e-30)
    w_quad = 1.0 / (lam - mu)
    w_mid = (T - ax) / (mu * ax_safe)
    w_tail = r / ax_safe + alpha * b / (ax_safe * ax_safe)
    w = jnp.where(ax <= x1, w_quad, jnp.where(ax < x2, w_mid, w_tail))
    if clip:
        lo, hi = WEIGHT_CLIP
        w = jnp.clip(w, lo, hi)
    return w
