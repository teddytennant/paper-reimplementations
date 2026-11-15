"""Dependence density ρ (Bernstein form, eq. 8)."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, xlogy


def bernstein_rho(u, iota, N: int):
    """ρ(u) = (N-1) Σ_{i=0}^{N-2} ι(i) Binom(N-2, i) u^i (1-u)^{N-2-i}.

    Parameters
    ----------
    u : array-like in [0, 1]
    iota : array-like of length N-1, values ι(0), …, ι(N-2)
    N : sequence length
    """
    u = jnp.asarray(u)
    iota = jnp.asarray(iota)
    n = N - 2
    i = jnp.arange(n + 1)
    log_c = gammaln(n + 1.0) - gammaln(i + 1.0) - gammaln(n - i + 1.0)
    u_exp = jnp.asarray(u)[..., None]
    i_f = i.astype(u_exp.dtype)
    n_f = jnp.asarray(n, dtype=u_exp.dtype)
    log_pmf = log_c.astype(u_exp.dtype) + xlogy(i_f, u_exp) + xlogy(n_f - i_f, 1.0 - u_exp)
    pmf = jnp.exp(log_pmf)
    expect = jnp.sum(pmf * iota.astype(pmf.dtype), axis=-1)
    return (N - 1) * expect


def binomial_mixture_rho(u, iota, N: int):
    """ρ(u) = (N-1) E[ι(B)] for B ~ Binomial(N-2, u).

    Uses the binomial recurrence for the mixture weights (a different
    numerical path from :func:`bernstein_rho`).
    """
    u_arr = np.asarray(u, dtype=np.float64)
    iota = np.asarray(iota, dtype=np.float64)
    scalar = u_arr.ndim == 0
    u_flat = np.atleast_1d(u_arr).ravel()
    n = N - 2
    pmf = np.zeros((u_flat.size, n + 1), dtype=np.float64)
    pmf[:, 0] = np.power(np.clip(1.0 - u_flat, 0.0, 1.0), n)
    for i in range(n):
        one_m = 1.0 - u_flat
        ratio = np.zeros_like(u_flat)
        nz = one_m > 0.0
        ratio[nz] = ((n - i) / (i + 1.0)) * (u_flat[nz] / one_m[nz])
        pmf[:, i + 1] = pmf[:, i] * ratio
    at_one = u_flat >= 1.0
    if np.any(at_one):
        pmf[at_one, :] = 0.0
        pmf[at_one, n] = 1.0
    expect = pmf @ iota
    out = (N - 1) * expect
    if scalar:
        return out[0]
    return out.reshape(np.asarray(u).shape)


def rho_from_callable(fn: Callable) -> Callable:
    """Wrap a scalar/vector callable as a numpy-valued density ρ."""

    def rho(u):
        u_np = np.asarray(u)
        out = fn(u_np)
        return np.asarray(out, dtype=np.float64)

    return rho
