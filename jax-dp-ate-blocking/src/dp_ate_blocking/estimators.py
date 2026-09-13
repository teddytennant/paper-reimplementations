"""Non-private ATE estimators and sensitivities from arXiv:2609.09536."""

from __future__ import annotations

import jax.numpy as jnp


def clip_propensity(e_hat, omega):
    """Clip propensity scores to [omega, 1 - omega] as in SeqIPW / Theorem 6."""
    omega = jnp.asarray(omega)
    e_hat = jnp.asarray(e_hat)
    return jnp.clip(e_hat, omega, 1.0 - omega)


def ipw(y, w, e_hat, omega=None):
    """Horvitz–Thompson IPW (paper §2.2).

    tau_IPW = (1/n) sum_i [ W_i Y_i / e(X_i) - (1-W_i) Y_i / (1-e(X_i)) ]
    """
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    e_hat = jnp.asarray(e_hat, dtype=y.dtype)
    if omega is not None:
        e_hat = clip_propensity(e_hat, omega)
    n = y.shape[0]
    treated = w * y / e_hat
    control = (1.0 - w) * y / (1.0 - e_hat)
    return jnp.sum(treated - control) / n


def assign_bins(e_hat, m):
    """Equal-spaced bins on [0, 1]: J_k = {(k-1)/m <= e < k/m}, e=1 in the last bin.

    Returns 0-based bin indices in {0, ..., m-1}.
    """
    m = int(m)
    e_hat = jnp.clip(jnp.asarray(e_hat), 0.0, 1.0)
    idx = jnp.floor(e_hat * m).astype(jnp.int32)
    return jnp.clip(idx, 0, m - 1)


def block_stats(y, w, bins, m):
    """Per-bin counts n_{k,w} and outcome sums for w in {0,1}."""
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    bins = jnp.asarray(bins)
    n_k1 = jnp.zeros((m,), dtype=y.dtype)
    n_k0 = jnp.zeros((m,), dtype=y.dtype)
    s_k1 = jnp.zeros((m,), dtype=y.dtype)
    s_k0 = jnp.zeros((m,), dtype=y.dtype)
    n_k1 = n_k1.at[bins].add(w)
    n_k0 = n_k0.at[bins].add(1.0 - w)
    s_k1 = s_k1.at[bins].add(w * y)
    s_k0 = s_k0.at[bins].add((1.0 - w) * y)
    return n_k0, n_k1, s_k0, s_k1


def bps(y, w, e_hat, m):
    """Blocking on the propensity score with equal-spaced bins (paper §2.2).

    Skips a block if n_{k,0}=0 or n_{k,1}=0. Divides by n, not by kept-bin size.
    """
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    e_hat = jnp.asarray(e_hat, dtype=y.dtype)
    m = int(m)
    n = y.shape[0]
    bins = assign_bins(e_hat, m)
    n_k0, n_k1, s_k0, s_k1 = block_stats(y, w, bins, m)
    n_k = n_k0 + n_k1
    valid = (n_k0 > 0) & (n_k1 > 0)
    tau_k = (n_k / n_k1) * s_k1 - (n_k / n_k0) * s_k0
    tau_k = jnp.where(valid, tau_k, 0.0)
    return jnp.sum(tau_k) / n


def aipw(y, w, e_hat, g0, g1, omega=None):
    """AIPW (Appendix A), non-private.

    tau_AIPW = (1/n) sum_i [ g1(X_i) - g0(X_i)
        + W_i (Y_i - g1(X_i)) / e(X_i)
        - (1-W_i) (Y_i - g0(X_i)) / (1-e(X_i)) ]
    """
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    e_hat = jnp.asarray(e_hat, dtype=y.dtype)
    g0 = jnp.asarray(g0, dtype=y.dtype)
    g1 = jnp.asarray(g1, dtype=y.dtype)
    if omega is not None:
        e_hat = clip_propensity(e_hat, omega)
    n = y.shape[0]
    term = (
        g1
        - g0
        + w * (y - g1) / e_hat
        - (1.0 - w) * (y - g0) / (1.0 - e_hat)
    )
    return jnp.sum(term) / n


def ipw_sensitivity(n, C_y, omega):
    """ℓ1 sensitivity of clipped IPW (Theorem 6): 2 C_y / (n ω)."""
    n = jnp.asarray(n)
    C_y = jnp.asarray(C_y)
    omega = jnp.asarray(omega)
    return 2.0 * C_y / (n * omega)


def aipw_sensitivity(n, C, omega):
    """ℓ1 sensitivity of AIPW (Theorem 10): (4 C / n) (1 + 1/ω)."""
    n = jnp.asarray(n)
    C = jnp.asarray(C)
    omega = jnp.asarray(omega)
    return (4.0 * C / n) * (1.0 + 1.0 / omega)


def difference_in_means(y, w):
    """Raw difference in means, for confounding checks."""
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    n1 = jnp.sum(w)
    n0 = jnp.sum(1.0 - w)
    mu1 = jnp.where(n1 > 0, jnp.sum(w * y) / n1, 0.0)
    mu0 = jnp.where(n0 > 0, jnp.sum((1.0 - w) * y) / n0, 0.0)
    return mu1 - mu0
