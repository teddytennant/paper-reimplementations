"""MSFC, structure-aware regularization, probes, and the joint objective."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from jax_gralod.ops import Array, antialiased_resize, box_filter


def mean_l1(pred: Array, target: Array) -> Array:
    return jnp.mean(jnp.abs(pred - target))


def reconstruction_loss(pred: Array, clean: Array) -> Array:
    """L_rec: L1. Paper follows the backbone; this package uses L1."""
    return mean_l1(pred, clean)


def resize_clean(clean: Array, height: int, width: int) -> Array:
    """x_s: clean target on the stage-s lattice (anti-aliased when reducing)."""
    return antialiased_resize(clean, height, width)


def local_error(xhat: Array, clean_s: Array, ksize: int = 5) -> Array:
    """e_{s,l}(p) = A_r(|xhat_s^l - x_s|)(p) (Eq. 7 / 22). Scalar per location."""
    abs_err = jnp.abs(xhat - clean_s)
    smoothed = box_filter(abs_err, ksize=ksize)
    return jnp.mean(smoothed, axis=-1)


def msfc_costs(errors: Array, gamma: float = 0.05) -> Array:
    """c_{s,l} = e_{s,l} + gamma * l / (L-1) (Eq. 8 / 23). `errors` is (..., L)."""
    n_levels = errors.shape[-1]
    levels = jnp.arange(n_levels, dtype=errors.dtype)
    denom = jnp.maximum(float(n_levels - 1), 1.0)
    return errors + gamma * levels / denom


def msfc_distribution(costs: Array, tau: float = 0.20) -> Array:
    """q_{s,l} proportional to exp(-c_{s,l}/tau) (Eq. 26)."""
    return jax.nn.softmax(-costs / tau, axis=-1)


def msfc_target(
    errors: Array,
    gamma: float = 0.05,
    tau: float = 0.20,
) -> tuple[Array, Array, Array]:
    """lambda*_s(p) = sum_l l q_{s,l}(p) (Eq. 27). Returns (lambda_star, q, c)."""
    costs = msfc_costs(errors, gamma=gamma)
    q = msfc_distribution(costs, tau=tau)
    n_levels = errors.shape[-1]
    levels = jnp.arange(n_levels, dtype=errors.dtype)
    lam_star = jnp.sum(q * levels, axis=-1)
    return lam_star, q, costs


def l_cal(lambdas: Sequence[Array], targets: Sequence[Array]) -> Array:
    """Eq. 9: mean |lambda_s - sg(lambda*_s)| over stages and Omega_s."""
    acc = 0.0
    n = len(lambdas)
    for lam, tgt in zip(lambdas, targets):
        lam_s = lam.reshape(lam.shape[0], lam.shape[1], lam.shape[2])
        tgt_s = tgt.reshape(lam_s.shape)
        acc = acc + jnp.mean(jnp.abs(lam_s - jax.lax.stop_gradient(tgt_s)))
    return acc / float(n)


def _affinity_l1(a: Array, b: Array, beta: float) -> Array:
    return jnp.exp(-beta * jnp.sum(jnp.abs(a - b), axis=-1))


def stage_structure_penalty(lam: Array, clean_s: Array, beta: float = 10.0) -> Array:
    """One-stage SAR: (1/|Omega|) sum_p sum_{q in N(p)} w_pq |lambda_p - lambda_q|.

    N(p) is 4-connected (no wrap, no self at the border). w_pq uses clean x_s
    (Eq. 10 / 30 / 31). Each interior edge is counted twice, once from each end.
    """
    if lam.ndim == 4:
        lam = lam[..., 0]
    b, h, w = lam.shape
    acc = jnp.zeros((), dtype=lam.dtype)
    # vertical: p=(i,j), q=(i+1,j) and the reverse
    w_v = _affinity_l1(clean_s[:, 1:], clean_s[:, :-1], beta)
    d_v = jnp.abs(lam[:, 1:] - lam[:, :-1])
    acc = acc + 2.0 * jnp.sum(w_v * d_v)
    # horizontal
    w_h = _affinity_l1(clean_s[:, :, 1:], clean_s[:, :, :-1], beta)
    d_h = jnp.abs(lam[:, :, 1:] - lam[:, :, :-1])
    acc = acc + 2.0 * jnp.sum(w_h * d_h)
    return acc / float(b * h * w)


def l_str(
    lambdas: Sequence[Array],
    cleans: Sequence[Array],
    beta: float = 10.0,
) -> Array:
    """Eq. 10 / 31, averaged over stages. Uses clean x_s, not the degraded image."""
    acc = 0.0
    n = len(lambdas)
    for lam, x_s in zip(lambdas, cleans):
        acc = acc + stage_structure_penalty(lam, x_s, beta=beta)
    return acc / float(n)


def l_probe(xhats: Sequence[Sequence[Array]], cleans: Sequence[Array]) -> Array:
    """Eq. 28: (1/S) sum_s (1/L) (1/|Omega_s|) sum_l ||xhat_s^l - x_s||_1.

    Implemented as mean absolute error over spatial locations and channels.
    """
    acc = 0.0
    n_stages = len(cleans)
    for s, x_s in enumerate(cleans):
        level_acc = 0.0
        n_levels = len(xhats[s])
        for xhat in xhats[s]:
            level_acc = level_acc + mean_l1(xhat, x_s)
        acc = acc + level_acc / float(n_levels)
    return acc / float(n_stages)


def total_loss(
    rec: Array,
    cal: Array,
    str_loss: Array,
    probe: Array,
    lambda_cal: float = 0.10,
    lambda_str: float = 0.05,
    lambda_probe: float = 0.10,
) -> Array:
    """Eq. 11: L = L_rec + lambda_cal L_cal + lambda_str L_str + lambda_probe L_probe."""
    return rec + lambda_cal * cal + lambda_str * str_loss + lambda_probe * probe


def calibration_ramp(
    step: Array | int,
    warmup_steps: int,
    lambda_cal_max: float = 0.10,
) -> Array:
    """Eq. 32: lambda_cal(t) = lambda_cal_max * r(t), r from 0 to 1 over warmup.

    If warmup_steps <= 0, r=1 (the default used in tests).
    """
    step_f = jnp.asarray(step, dtype=jnp.float32)
    if warmup_steps <= 0:
        r = jnp.ones_like(step_f)
    else:
        r = jnp.clip(step_f / float(warmup_steps), 0.0, 1.0)
    return jnp.asarray(lambda_cal_max, dtype=jnp.float32) * r


def stack_level_errors(
    xhats: Sequence[Array],
    clean_s: Array,
    ksize: int = 5,
) -> Array:
    """Stack e_{s,l} over l into (B, H, W, L)."""
    errs = [local_error(xhat, clean_s, ksize=ksize) for xhat in xhats]
    return jnp.stack(errs, axis=-1)
