"""Laplace mechanism, SeqIPW, and DPBlocking from arXiv:2609.09536."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .estimators import assign_bins, block_stats, ipw
from .propensity import fit_propensity, predict_propensity


def sample_laplace(key, shape, scale):
    """Laplace(0, b) via inverse CDF: U ~ Unif(-1/2, 1/2), X = -b sgn(U) ln(1-2|U|)."""
    scale = jnp.asarray(scale)
    u = jax.random.uniform(
        key, shape, minval=-0.5, maxval=0.5, dtype=jnp.result_type(scale, jnp.float64)
    )
    u = jnp.clip(u, -0.5 + 1e-15, 0.5 - 1e-15)
    return -scale * jnp.sign(u) * jnp.log1p(-2.0 * jnp.abs(u))


def default_m(n, eps):
    """Number of equal-spaced bins (paper §4.2).

    m = max{3, floor( (n/1000 + (eps-0.25)/0.25) / 2 )}
    then cap at max(5, 20 eps).
    """
    n = float(n)
    eps = float(eps)
    inner = (n / 1000.0 + (eps - 0.25) / 0.25) / 2.0
    m = max(3, int(jnp.floor(inner)))
    cap = max(5.0, 20.0 * eps)
    return max(1, int(min(m, cap)))


def seq_ipw(
    x,
    w,
    y,
    key=None,
    eps=1.0,
    omega=0.1,
    alpha=0.2,
    C_y=1.5,
    e_hat=None,
    add_noise=True,
):
    """Algorithm 1 SeqIPW.

    Train an (α ε)-DP propensity model, compute clipped IPW, add
    Lap(2 C_y / (n ω ε2)) with ε2 = (1-α)ε.
    Defaults: α=0.2, ω=0.1 (paper §4.2).
    """
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    x = jnp.asarray(x)
    n = y.shape[0]
    eps1 = alpha * eps
    eps2 = (1.0 - alpha) * eps

    if e_hat is None:
        if key is None:
            raise ValueError("seq_ipw requires a PRNG key unless e_hat is provided")
        key, k_fit = jax.random.split(key)
        theta = fit_propensity(x, w, key=k_fit, eps=eps1, add_noise=add_noise)
        e_hat = predict_propensity(theta, x)
    e_hat = jnp.asarray(e_hat, dtype=y.dtype)
    tau = ipw(y, w, e_hat, omega=omega)

    if (not add_noise) or (key is None):
        return tau
    scale = 2.0 * C_y / (n * omega * eps2)
    scale = jnp.where(jnp.isfinite(scale), scale, 0.0)
    return tau + sample_laplace(key, (), scale)


def dp_blocking(
    x,
    w,
    y,
    key=None,
    eps=1.0,
    C_y=1.5,
    alpha1=0.1,
    alpha2=0.35,
    alpha3=0.55,
    m=None,
    T=1.5,
    e_hat=None,
    add_noise=True,
):
    """Algorithm 2 DPBlocking.

    Defaults: T=1.5, α1=0.1, α2=0.35, α3=0.55. If ñ == 0, return 0.0.
    """
    y = jnp.asarray(y)
    w = jnp.asarray(w, dtype=y.dtype)
    x = jnp.asarray(x)
    n = int(y.shape[0])
    if m is None:
        m = default_m(n, eps)
    m = int(m)

    eps1 = alpha1 * eps
    eps2 = alpha2 * eps
    eps3 = alpha3 * eps

    if e_hat is None:
        if key is None:
            raise ValueError("dp_blocking requires a PRNG key unless e_hat is provided")
        key, k_fit = jax.random.split(key)
        theta = fit_propensity(x, w, key=k_fit, eps=eps1, add_noise=add_noise)
        e_hat = predict_propensity(theta, x)
    e_hat = jnp.asarray(e_hat, dtype=y.dtype)

    bins = assign_bins(e_hat, m)
    n_k0, n_k1, s_k0, s_k1 = block_stats(y, w, bins, m)

    if add_noise and key is not None:
        key, k0, k1, k2, k3 = jax.random.split(key, 5)
        scale_n = 2.0 / eps2
        scale_s = 2.0 * C_y / eps3
        scale_n = jnp.where(jnp.isfinite(scale_n), scale_n, 0.0)
        scale_s = jnp.where(jnp.isfinite(scale_s), scale_s, 0.0)
        n_k0 = n_k0 + sample_laplace(k0, (m,), scale_n)
        n_k1 = n_k1 + sample_laplace(k1, (m,), scale_n)
        s_k0 = s_k0 + sample_laplace(k2, (m,), scale_s)
        s_k1 = s_k1 + sample_laplace(k3, (m,), scale_s)

    # Drop if noisy count < T (paper). Also skip exact zeros so empty
    # treatment cells match BPS when T <= 0 and Laplace noise is off.
    keep = (n_k0 >= T) & (n_k1 >= T) & (n_k0 != 0) & (n_k1 != 0)
    tau_k = (n_k1 + n_k0) * (s_k1 / n_k1 - s_k0 / n_k0)
    tau_k = jnp.where(keep, tau_k, 0.0)
    n_tilde = jnp.sum(jnp.where(keep, n_k0 + n_k1, 0.0))
    tau = jnp.sum(tau_k) / n_tilde
    return jnp.where(n_tilde == 0, jnp.asarray(0.0, dtype=y.dtype), tau)
