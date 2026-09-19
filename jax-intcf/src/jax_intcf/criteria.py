"""Splitting criteria from arXiv:2609.16971 (IntCT / IntCF).

Equation numbers follow the paper. N is the full training (or, in the
validation section, validation) sample size, not the parent node size.
"""

from __future__ import annotations

from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

Array = Any


def _f(x: Array) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def _safe_div(num: Array, den: Array) -> Array:
    num = _f(num)
    den = _f(den)
    return jnp.where(den != 0.0, num / den, jnp.zeros_like(num, dtype=jnp.float64))


def difference_in_means(y: Array, t: Array) -> Array:
    """Eq 10: tau_hat_e = mean(y | t=1, e) - mean(y | t=0, e)."""
    y = _f(y)
    t = _f(t)
    n1 = jnp.sum(t)
    n0 = jnp.sum(1.0 - t)
    m1 = _safe_div(jnp.sum(y * t), n1)
    m0 = _safe_div(jnp.sum(y * (1.0 - t)), n0)
    valid = (n1 > 0.0) & (n0 > 0.0)
    return jnp.where(valid, m1 - m0, jnp.nan)


def het_criterion(n_L: Array, n_R: Array, N: Array, tau_L: Array, tau_R: Array) -> Array:
    """Eq 11: Het_hat = (nL nR) / (N (nL + nR)) * (tau_L - tau_R)^2."""
    n_L, n_R, N = _f(n_L), _f(n_R), _f(N)
    tau_L, tau_R = _f(tau_L), _f(tau_R)
    return _safe_div(n_L * n_R, N * (n_L + n_R)) * (tau_L - tau_R) ** 2


def bias_criterion(
    n_L: Array,
    n_R: Array,
    N: Array,
    tau_parent: Array,
    tau_L: Array,
    tau_R: Array,
) -> Array:
    """Eq 15: Bias_hat = (nL + nR)/N * (tau_parent - weighted child mean)^2."""
    n_L, n_R, N = _f(n_L), _f(n_R), _f(N)
    tau_parent, tau_L, tau_R = _f(tau_parent), _f(tau_L), _f(tau_R)
    child_mean = _safe_div(n_L * tau_L + n_R * tau_R, n_L + n_R)
    return _safe_div(n_L + n_R, N) * (tau_parent - child_mean) ** 2


def mean_squared_change(
    n_L: Array,
    n_R: Array,
    N: Array,
    tau_parent: Array,
    tau_L: Array,
    tau_R: Array,
) -> Array:
    """Eq 14: (1/N) [ nL (tau_parent - tau_L)^2 + nR (tau_parent - tau_R)^2 ]."""
    n_L, n_R, N = _f(n_L), _f(n_R), _f(N)
    tau_parent, tau_L, tau_R = _f(tau_parent), _f(tau_L), _f(tau_R)
    return _safe_div(
        n_L * (tau_parent - tau_L) ** 2 + n_R * (tau_parent - tau_R) ** 2,
        N,
    )


def intcf_delta(het: Array, bias: Array) -> Array:
    """Eq 17: Delta_IntCF(L, R) = max(Het_hat, Bias_hat)."""
    return jnp.maximum(_f(het), _f(bias))


def signed_het(n_L: Array, n_R: Array, tau_L: Array, tau_R: Array) -> Array:
    """Eq 18: Het^± = sqrt(nL nR)/(nL + nR) * (tau_L - tau_R)."""
    n_L, n_R = _f(n_L), _f(n_R)
    return _safe_div(jnp.sqrt(n_L * n_R), n_L + n_R) * (_f(tau_L) - _f(tau_R))


def signed_bias(
    tau_parent: Array,
    n_L: Array,
    n_R: Array,
    tau_L: Array,
    tau_R: Array,
) -> Array:
    """Eq 19: Bias^± = tau_parent - (nL tau_L + nR tau_R)/(nL + nR)."""
    n_L, n_R = _f(n_L), _f(n_R)
    child_mean = _safe_div(_f(n_L) * _f(tau_L) + _f(n_R) * _f(tau_R), n_L + n_R)
    return _f(tau_parent) - child_mean


def unsigned_from_signed(signed: Array, n_L: Array, n_R: Array, N: Array) -> Array:
    """Squaring and multiplying by (nL + nR)/N recovers Het and Bias."""
    return _safe_div(_f(n_L) + _f(n_R), _f(N)) * _f(signed) ** 2


def het_validation(n_L: Array, n_R: Array, N: Array, tau_L_b: Array, tau_R_b: Array) -> Array:
    """Eq 22: Het^v using backward child predictions."""
    return het_criterion(n_L, n_R, N, tau_L_b, tau_R_b)


def bias_validation(
    n_L: Array,
    n_R: Array,
    N: Array,
    tau_parent_f: Array,
    tau_parent_b: Array,
    tau_L_f: Array,
    tau_L_b: Array,
    tau_R_f: Array,
    tau_R_b: Array,
) -> Array:
    """Eq 23: Bias^v from forward/backward parent and child predictions."""
    n_L, n_R, N = _f(n_L), _f(n_R), _f(N)
    term_p = _safe_div(n_L + n_R, N) * (_f(tau_parent_f) - _f(tau_parent_b)) ** 2
    term_L = _safe_div(n_L, N) * (_f(tau_L_f) - _f(tau_L_b)) ** 2
    term_R = _safe_div(n_R, N) * (_f(tau_R_f) - _f(tau_R_b)) ** 2
    return term_p - term_L - term_R


def lemma4_mse_decrease(
    tau_parent_f: Array,
    tau_L_f: Array,
    tau_R_f: Array,
    tau_i_parent: Array,
    tau_i_L: Array,
    tau_i_R: Array,
    N: Array,
) -> Array:
    """Lemma 4: MSE drop from replacing parent forward ATE by leaf predictions,
    net of the same drop measured at the children.
    """
    N = _f(N)
    parent = jnp.sum((_f(tau_parent_f) - _f(tau_i_parent)) ** 2)
    left = jnp.sum((_f(tau_L_f) - _f(tau_i_L)) ** 2)
    right = jnp.sum((_f(tau_R_f) - _f(tau_i_R)) ** 2)
    return _safe_div(parent - left - right, N)


def evaluate_split(
    y: Array,
    t: Array,
    left_mask: Array,
    N: float | None = None,
    tau_parent: Array | None = None,
) -> dict[str, Array]:
    """Difference-in-means, Het, Bias, Eq 16 residual, signed terms, IntCF delta."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    left_mask = np.asarray(left_mask, dtype=bool).reshape(-1)
    if N is None:
        N = float(y.shape[0])
    else:
        N = float(N)
    if tau_parent is None:
        tau_parent = difference_in_means(y, t)
    y_L, t_L = y[left_mask], t[left_mask]
    y_R, t_R = y[~left_mask], t[~left_mask]
    n_L = float(y_L.shape[0])
    n_R = float(y_R.shape[0])
    tau_L = difference_in_means(y_L, t_L)
    tau_R = difference_in_means(y_R, t_R)
    het = het_criterion(n_L, n_R, N, tau_L, tau_R)
    bias = bias_criterion(n_L, n_R, N, tau_parent, tau_L, tau_R)
    mse = mean_squared_change(n_L, n_R, N, tau_parent, tau_L, tau_R)
    het_pm = signed_het(n_L, n_R, tau_L, tau_R)
    bias_pm = signed_bias(tau_parent, n_L, n_R, tau_L, tau_R)
    return {
        "n_L": n_L,
        "n_R": n_R,
        "N": N,
        "tau_parent": tau_parent,
        "tau_L": tau_L,
        "tau_R": tau_R,
        "het": het,
        "bias": bias,
        "mse": mse,
        "delta": intcf_delta(het, bias),
        "signed_het": het_pm,
        "signed_bias": bias_pm,
    }


def scale_importances(raw: Array) -> np.ndarray:
    """Scale so the non-negative scores sum to 1 (bias scores may be negative)."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    pos = float(np.maximum(raw, 0.0).sum())
    if pos <= 0.0:
        return np.zeros_like(raw)
    return raw / pos
