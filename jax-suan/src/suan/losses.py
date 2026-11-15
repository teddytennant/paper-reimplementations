"""Suan losses and closed-form gradient coefficients (arXiv:2609.08634).

Equations follow Corollary 3.3 / Definition 3.4 and Propositions 3.1–3.2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

DEFAULT_TAU = 1.0
DEFAULT_BETA = 0.1


def log_cosh(z: jax.Array) -> jax.Array:
    """Stable log(cosh(z))."""
    az = jnp.abs(z)
    return az + jax.nn.softplus(-2.0 * az) - jnp.log(2.0)


def preference_loss(
    log_pi_plus: jax.Array,
    log_pi_minus: jax.Array,
    tau: float = DEFAULT_TAU,
) -> jax.Array:
    """L_P, Corollary 3.3 / Eq (10), per example.

    L_P(θ) = −log σ( log π_θ(y+|x) − log π_θ(y−|x) + τ )
    """
    t = log_pi_plus - log_pi_minus + tau
    return jax.nn.softplus(-t)


def regularizer_loss(
    log_pi_plus: jax.Array,
    log_pi_minus: jax.Array,
    log_ref_plus: jax.Array,
    log_ref_minus: jax.Array,
) -> jax.Array:
    """L_R, Corollary 3.3 / Eq (11), per example.

    L_R(θ) = Σ_{y ∈ {y+, y−}} log cosh( (1/2) log(π_θ(y|x)/π_ref(y|x)) )
    """
    term_plus = log_cosh(0.5 * (log_pi_plus - log_ref_plus))
    term_minus = log_cosh(0.5 * (log_pi_minus - log_ref_minus))
    return term_plus + term_minus


def suan_loss(
    log_pi_plus: jax.Array,
    log_pi_minus: jax.Array,
    log_ref_plus: jax.Array,
    log_ref_minus: jax.Array,
    tau: float = DEFAULT_TAU,
    beta: float = DEFAULT_BETA,
) -> jax.Array:
    """L_Suan, Definition 3.4 / Eq (12): L_P + β L_R, per example."""
    return preference_loss(log_pi_plus, log_pi_minus, tau=tau) + beta * regularizer_loss(
        log_pi_plus, log_pi_minus, log_ref_plus, log_ref_minus
    )


def preference_gradient_coefficient(
    log_pi_plus: jax.Array,
    log_pi_minus: jax.Array,
    tau: float = DEFAULT_TAU,
) -> jax.Array:
    """Scalar coefficient in Proposition 3.1 / Eq (6).

    −σ( log(π_θ(y−|x) / π_θ(y+|x)) − τ )
    """
    return -jax.nn.sigmoid(log_pi_minus - log_pi_plus - tau)


def regularizer_gradient_coefficient(
    log_pi: jax.Array,
    log_ref: jax.Array,
) -> jax.Array:
    """Coefficient in Proposition 3.2 / Eq (8): σ(log(π_θ/π_ref)) − 1/2."""
    return jax.nn.sigmoid(log_pi - log_ref) - 0.5


def preference_closed_form_dlog(
    log_pi_plus: jax.Array,
    log_pi_minus: jax.Array,
    tau: float = DEFAULT_TAU,
) -> tuple[jax.Array, jax.Array]:
    """∂L_P / ∂log π(y+), ∂L_P / ∂log π(y−) from Proposition 3.1."""
    coeff = preference_gradient_coefficient(log_pi_plus, log_pi_minus, tau=tau)
    return coeff, -coeff


def regularizer_closed_form_dlog(
    log_pi_plus: jax.Array,
    log_pi_minus: jax.Array,
    log_ref_plus: jax.Array,
    log_ref_minus: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """∂L_R / ∂log π(y+), ∂L_R / ∂log π(y−) from Proposition 3.2."""
    d_plus = regularizer_gradient_coefficient(log_pi_plus, log_ref_plus)
    d_minus = regularizer_gradient_coefficient(log_pi_minus, log_ref_minus)
    return d_plus, d_minus
