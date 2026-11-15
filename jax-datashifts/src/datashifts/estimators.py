"""Plug-in / debiased X-shift (eq. 8–9) and γ*-concept-shift (eq. 11)."""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from .sinkhorn import Array, CostFn, as_points, entropic_ot


def plugin_w_beta(
    x_s: Array,
    x_t: Array,
    beta: float = 0.01,
    rho: Optional[CostFn] = None,
    n_iter: int = 2000,
    return_coupling: bool = False,
):
    """Plug-in W_β on the full samples (eq. 8)."""
    w_beta, gamma = entropic_ot(x_s, x_t, beta=beta, rho=rho, n_iter=n_iter)
    if return_coupling:
        return w_beta, gamma
    return w_beta


def _half_split(x: Array) -> Tuple[Array, Array]:
    x = as_points(x)
    n = x.shape[0]
    n_half = n // 2
    if n_half < 1:
        raise ValueError("need at least 2 samples per domain for the debiased estimator")
    return x[:n_half], x[n_half : 2 * n_half]


def debiased_w_beta(
    x_s: Array,
    x_t: Array,
    beta: float = 0.01,
    rho: Optional[CostFn] = None,
    n_iter: int = 2000,
) -> Array:
    """Debiased X-shift estimator W_β^deb (Def. 4.1 / eq. 9)."""
    s1, s2 = _half_split(x_s)
    t1, t2 = _half_split(x_t)
    w_st1, _ = entropic_ot(s1, t1, beta=beta, rho=rho, n_iter=n_iter)
    w_st2, _ = entropic_ot(s2, t2, beta=beta, rho=rho, n_iter=n_iter)
    w_ss, _ = entropic_ot(s1, s2, beta=beta, rho=rho, n_iter=n_iter)
    w_tt, _ = entropic_ot(t1, t2, beta=beta, rho=rho, n_iter=n_iter)
    inside = 0.5 * w_st1**2 + 0.5 * w_st2**2 - 0.5 * w_ss**2 - 0.5 * w_tt**2
    return jnp.sqrt(jnp.abs(inside))


def _label_cost(y_s: Array, y_t: Array, rho_y: Optional[CostFn]) -> Array:
    ys = as_points(y_s)
    yt = as_points(y_t)
    if rho_y is not None:
        return jnp.asarray(rho_y(ys, yt))
    # |·| for scalar labels; Euclidean otherwise (Def. 4.3).
    if ys.shape[-1] == 1 and yt.shape[-1] == 1:
        return jnp.abs(ys - yt.T)
    return jnp.linalg.norm(ys[:, None, :] - yt[None, :, :], axis=-1)


def concept_shift(
    x_s: Array,
    y_s: Array,
    x_t: Array,
    y_t: Array,
    beta: float = 0.01,
    rho_x: Optional[CostFn] = None,
    rho_y: Optional[CostFn] = None,
    n_iter: int = 2000,
    gamma: Optional[Array] = None,
) -> Array:
    """γ*-concept-shift estimator Ŝ_Cpt (Def. 4.3 / eq. 11).

    γ̂* is the coupling of the plug-in W_β on the full covariate samples.
    """
    if gamma is None:
        _, gamma = entropic_ot(x_s, x_t, beta=beta, rho=rho_x, n_iter=n_iter)
    c_y = _label_cost(y_s, y_t, rho_y)
    return jnp.sum(c_y * gamma)
