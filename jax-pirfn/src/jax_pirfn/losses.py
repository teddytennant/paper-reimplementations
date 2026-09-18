"""Physics-informed residual + BC/IC loss with L2 (and optional L1) on c."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

Array = jax.Array

# Small default; the paper uses lambda_1 = lambda_2 = 1 on un-normalized sums.
DEFAULT_L2 = 1e-6


def mean_sq(r: Array) -> Array:
    return jnp.mean(r ** 2)


def pirfn_loss(
    residual: Array,
    bc_residual: Array,
    c: Array,
    ic_residual: Optional[Array] = None,
    lam_l2: float = DEFAULT_L2,
    lam_l1: float = 0.0,
) -> Array:
    """L = mean N[u]^2 + mean BC^2 [+ mean IC^2] + lam_l2 ||c||_2^2 + lam_l1 ||c||_1."""
    loss = mean_sq(residual) + mean_sq(bc_residual)
    if ic_residual is not None:
        loss = loss + mean_sq(ic_residual)
    loss = loss + lam_l2 * jnp.sum(c ** 2) + lam_l1 * jnp.sum(jnp.abs(c))
    return loss


def loss_terms(
    residual: Array,
    bc_residual: Array,
    c: Array,
    ic_residual: Optional[Array] = None,
    lam_l2: float = DEFAULT_L2,
    lam_l1: float = 0.0,
) -> dict:
    terms = {
        "pde": mean_sq(residual),
        "bc": mean_sq(bc_residual),
        "l2": lam_l2 * jnp.sum(c ** 2),
        "l1": lam_l1 * jnp.sum(jnp.abs(c)),
    }
    if ic_residual is not None:
        terms["ic"] = mean_sq(ic_residual)
    terms["total"] = pirfn_loss(
        residual, bc_residual, c, ic_residual=ic_residual, lam_l2=lam_l2, lam_l1=lam_l1
    )
    return terms


def relative_l2(u_hat: Array, u_true: Array) -> Array:
    denom = jnp.linalg.norm(u_true)
    return jnp.linalg.norm(u_hat - u_true) / jnp.maximum(denom, 1e-12)
