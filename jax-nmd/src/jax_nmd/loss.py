"""Observable losses (eq 5-6, appendix B.4 eq 14-15)."""

from __future__ import annotations

import jax.numpy as jnp

from jax_nmd.config import NMDConfig


def ell_gauge(a, eps=1e-6):
    """ℓ_gauge(a) = 1 - Re(a) / (|a| + ε), eq 5 / appendix B.4 eq 15."""
    a = jnp.asarray(a)
    return 1.0 - jnp.real(a) / (jnp.abs(a) + eps)


def gauge_loss(left, right, is_anchor, eps=1e-6):
    """Mean of ℓ_gauge(l_{k,p}) and ℓ_gauge(r_{k,p}) over modes and anchor ports.

    L_gauge = 0 when no queried port is on the anchor (eq 15, |P_anchor|=0).
    left, right: (..., K, N) complex.
    is_anchor: (..., N) bool or 0/1.
    """
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    mask = jnp.asarray(is_anchor).astype(left.real.dtype)
    k_poles = left.shape[-2]
    pair = ell_gauge(left, eps=eps) + ell_gauge(right, eps=eps)
    masked = pair * mask[..., None, :]
    n_anchor = jnp.sum(mask, axis=-1)
    total = jnp.sum(masked, axis=(-2, -1))
    denom = (2.0 * k_poles) * jnp.maximum(n_anchor, 1.0)
    per = jnp.where(n_anchor > 0, total / denom, jnp.zeros_like(total))
    return jnp.mean(per)


def scattering_mae(s_pred, s_gt):
    """Per-entry MAE on S: mean of |Re diff| and |Im diff| over all entries.

    This is ordinary MAE on the real and imaginary parts stacked together,
    not the mean of the complex modulus |Spred - Sgt|.
    """
    diff = jnp.asarray(s_pred) - jnp.asarray(s_gt)
    return 0.5 * (
        jnp.mean(jnp.abs(jnp.real(diff))) + jnp.mean(jnp.abs(jnp.imag(diff)))
    )


def log_z_loss(z_pred, z_gt, eps=1e-2):
    """L_{Z,log} = mean |log(|Zpred|+ε) - log(|Zgt|+ε)|, ε = 1e-2."""
    z_pred = jnp.asarray(z_pred)
    z_gt = jnp.asarray(z_gt)
    lp = jnp.log(jnp.abs(z_pred) + eps)
    lg = jnp.log(jnp.abs(z_gt) + eps)
    return jnp.mean(jnp.abs(lp - lg))


def total_loss(
    s_pred,
    s_gt,
    z_pred,
    z_gt,
    left,
    right,
    is_anchor,
    lambda_z=0.1,
    lambda_g=0.5,
    eps_z=1e-2,
    eps_anchor=1e-6,
):
    """L = L_S + λ_Z L_{Z,log} + λ_g L_gauge, eq 6 / B.4 eq 14.

    Defaults λ_Z = 0.1, λ_g = 0.5 as in the paper.
    """
    ls = scattering_mae(s_pred, s_gt)
    lz = log_z_loss(z_pred, z_gt, eps=eps_z)
    lg = gauge_loss(left, right, is_anchor, eps=eps_anchor)
    loss = ls + lambda_z * lz + lambda_g * lg
    return loss, {"L_S": ls, "L_Z_log": lz, "L_gauge": lg}


def nmd_total_loss(s_pred, s_gt, z_pred, z_gt, left, right, is_anchor, cfg: NMDConfig):
    return total_loss(
        s_pred,
        s_gt,
        z_pred,
        z_gt,
        left,
        right,
        is_anchor,
        lambda_z=cfg.lambda_z,
        lambda_g=cfg.lambda_g,
        eps_z=cfg.eps_z,
        eps_anchor=cfg.eps_anchor,
    )
