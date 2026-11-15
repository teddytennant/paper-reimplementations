"""Validity, feasibility %, sparsity, proximity (paper §5.4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fcx.constraints import binary_feasible, unary_feasible


def validity_rate(pred_cf: jax.Array, y_target: jax.Array) -> jax.Array:
    """Fraction of CFs whose predicted class equals the target class."""
    return jnp.mean(pred_cf.astype(jnp.int32) == y_target.astype(jnp.int32))


def feasibility_rate(
    x: jax.Array,
    x_cf: jax.Array,
    binary: bool = False,
    age_idx: int = 0,
    edu_idx: int = 1,
    atol: float = 1e-5,
) -> jax.Array:
    """Fraction of CFs that satisfy the unary (Eq 5) or binary (Eqs 5–6) hard constraints."""
    if binary:
        ok = binary_feasible(x, x_cf, age_idx=age_idx, edu_idx=edu_idx, atol=atol)
    else:
        ok = unary_feasible(x, x_cf, age_idx=age_idx, atol=atol)
    return jnp.mean(ok.astype(jnp.float32))


def sparsity_score(x: jax.Array, x_cf: jax.Array, thresh: float = 0.01) -> jax.Array:
    """Mean number of features that change by more than thresh."""
    changed = jnp.abs(x_cf - x) > thresh
    return jnp.mean(jnp.sum(changed.astype(jnp.float32), axis=1))


def proximity(x: jax.Array, x_cf: jax.Array) -> jax.Array:
    """Mean L1 distance (paper tables report the negative of this)."""
    return jnp.mean(jnp.sum(jnp.abs(x_cf - x), axis=1))


def signed_proximity(x: jax.Array, x_cf: jax.Array) -> jax.Array:
    return -proximity(x, x_cf)
