"""Hard unary / binary feasibility penalties (paper §4.2, Eqs 5–6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def unary_penalty(x: jax.Array, x_cf: jax.Array, age_idx: int = 0) -> jax.Array:
    """FCx-U: max(0, x_age - x_age_cf). Penalises decreasing age."""
    return jnp.maximum(0.0, x[:, age_idx] - x_cf[:, age_idx])


def binary_penalty(
    x: jax.Array,
    x_cf: jax.Array,
    age_idx: int = 0,
    edu_idx: int = 1,
    alpha: float = 0.0,
    beta: float = 1.0,
) -> jax.Array:
    """FCx-B §4.2: max(0, (x_q - x_q_cf) - α - β(x_p - x_p_cf) - min(0, β)).

    q = age, p = education. With α=0, β=1 this is max(0, Δedu - Δage), i.e.
    education-up must be matched by at least as large an age-up.
    """
    dq = x[:, age_idx] - x_cf[:, age_idx]
    dp = x[:, edu_idx] - x_cf[:, edu_idx]
    return jnp.maximum(0.0, dq - alpha - beta * dp - jnp.minimum(jnp.asarray(beta, dtype=x.dtype), 0.0))


def feasibility_loss(
    x: jax.Array,
    x_cf: jax.Array,
    binary: bool = False,
    age_idx: int = 0,
    edu_idx: int = 1,
    alpha: float = 0.0,
    beta: float = 1.0,
) -> jax.Array:
    """Mean hard-constraint penalty. Both forms are computed; one is selected."""
    u = unary_penalty(x, x_cf, age_idx=age_idx)
    b = binary_penalty(x, x_cf, age_idx=age_idx, edu_idx=edu_idx, alpha=alpha, beta=beta)
    # keep both arithmetic branches; select with a float mix (binary is static in train)
    mix = jnp.asarray(1.0 if binary else 0.0, dtype=x.dtype)
    return jnp.mean((1.0 - mix) * u + mix * b)


def unary_feasible(x: jax.Array, x_cf: jax.Array, age_idx: int = 0, atol: float = 1e-5) -> jax.Array:
    """Eq 5: age_cf >= age."""
    return x_cf[:, age_idx] >= x[:, age_idx] - atol


def binary_feasible(
    x: jax.Array,
    x_cf: jax.Array,
    age_idx: int = 0,
    edu_idx: int = 1,
    atol: float = 1e-5,
) -> jax.Array:
    """Eqs 5–6: age never decreases; if education increases then age strictly increases."""
    age_ok = unary_feasible(x, x_cf, age_idx=age_idx, atol=atol)
    edu_up = x_cf[:, edu_idx] > x[:, edu_idx] + atol
    age_up = x_cf[:, age_idx] > x[:, age_idx] + atol
    return age_ok & (~edu_up | age_up)
