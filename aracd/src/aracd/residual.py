"""Ability residual correction (paper eqs. 7, 10–21)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import nn

from aracd.ops import layer_norm, logit, sigmoid

GAMMA_SI_DEFAULT = 1.5


def stu_item_residual(b_s, b_i, u_s, v_i, gamma_si=GAMMA_SI_DEFAULT):
    """Eqs. 11–12: r^{stu-item} = b_s + b_i + γ_SI <u_s, v_i> / sqrt(d).

    u_s, v_i: (..., d). b_s, b_i: (...) or broadcastable scalars.
    """
    rank = u_s.shape[-1]
    interaction = gamma_si * jnp.sum(u_s * v_i, axis=-1) / jnp.sqrt(jnp.asarray(rank, dtype=u_s.dtype))
    return b_s + b_i + interaction


def attention_query(e_s_att, v_i_att, eps: float = 1e-5):
    """Eq. 15: q_si^{att} = LN(tanh(e_s^{att} + v_i^{att})).

    LN is last-axis population LayerNorm with no affine parameters.
    """
    return layer_norm(jnp.tanh(e_s_att + v_i_att), eps=eps)


def attention_scores(q_si_att, k_k_att):
    """Eq. 17: e_sik = (q_si^{att} · k_k^{att}) / sqrt(d_att).

    q_si_att: (..., d_att)
    k_k_att: (K, d_att)
    returns: (..., K)
    """
    d_att = q_si_att.shape[-1]
    return (q_si_att @ jnp.swapaxes(k_k_att, -1, -2)) / jnp.sqrt(
        jnp.asarray(d_att, dtype=q_si_att.dtype)
    )


def q_masked_softmax(scores, q_i):
    """Eqs. 18–19: α_sik = softmax_k(e_sik + m_ik), m_ik = 0 if q_ik=1 else −∞.

    scores, q_i: (..., K)
    """
    masked = jnp.where(q_i > 0, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype))
    return nn.softmax(masked, axis=-1)


def concept_residual(alpha, b_k, b_s_k):
    """Eqs. 14, 21: r^{att} = sum_k α_sik (b_k + b_{s,k}).

    alpha: (..., K), b_k: (K,), b_s_k: (..., K)
    """
    values = b_k + b_s_k
    return jnp.sum(alpha * values, axis=-1)


def ability_residual(r_stu_item, r_concept, w_si, w_c):
    """Eq. 10: r^{ability} = w_SI r^{stu-item} + w_C r^{concept}."""
    return w_si * r_stu_item + w_c * r_concept


def ability_correct(y_base, r_ability, eps: float = 1e-6):
    """Eq. 7: z^{ability} = logit(y^{base}) + r^{ability}, y* = σ(z).

    y_base is clipped away from {0, 1} before the logit.
    """
    z = logit(y_base, eps=eps) + r_ability
    return sigmoid(z)
