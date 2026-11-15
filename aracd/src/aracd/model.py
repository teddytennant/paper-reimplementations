"""Full ARACD forward from student/item ids (NCDM + residual + affect)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from aracd.affect import affect_mlp, guess_slip, modulate
from aracd.ncdm import ncdm_base
from aracd.ops import LN_EPS, PROB_EPS, sigmoid
from aracd.residual import (
    GAMMA_SI_DEFAULT,
    ability_correct,
    ability_residual,
    attention_query,
    attention_scores,
    concept_residual,
    q_masked_softmax,
    stu_item_residual,
)


@dataclass(frozen=True)
class ARACDConfig:
    n_students: int
    n_items: int
    n_concepts: int
    rank: int = 16
    d_att: int = 16
    d_aff: int = 8
    mlp_hidden: int = 16
    affect_hidden: int = 8
    gamma_si: float = GAMMA_SI_DEFAULT
    ln_eps: float = LN_EPS
    prob_eps: float = PROB_EPS
    init_scale: float = 0.02


def init_params(key, n_students, n_items, n_concepts, *, config: ARACDConfig | None = None, **overrides):
    """Initialize residual biases at 0 and remaining tables with small noise.

    Paper: b terms start at 0 so the model degenerates to F_CD; low-rank and
    attention vectors get small random values. w_SI and w_C start at 1.
    NCDM table entries are unconstrained; forward applies sigmoid.
    """
    if config is None:
        config = ARACDConfig(n_students, n_items, n_concepts, **overrides)
    S, I, K = config.n_students, config.n_items, config.n_concepts
    H = config.mlp_hidden
    Ha = config.affect_hidden
    scale = config.init_scale
    keys = jax.random.split(key, 14)

    def noise(k, shape):
        return jax.random.normal(k, shape) * scale

    params = {
        "ncdm": {
            "student_mastery": noise(keys[0], (S, K)),
            "item_diff": noise(keys[1], (I, K)),
            "item_disc": noise(keys[2], (I, K)),
            "mlp_w1": noise(keys[3], (K, H)),
            "mlp_b1": jnp.zeros((H,)),
            "mlp_w2": noise(keys[4], (H,)),
            "mlp_b2": jnp.zeros(()),
        },
        "residual": {
            "b_s": jnp.zeros((S,)),
            "b_i": jnp.zeros((I,)),
            "u_s": noise(keys[5], (S, config.rank)),
            "v_i": noise(keys[6], (I, config.rank)),
            "w_si": jnp.ones(()),
            "w_c": jnp.ones(()),
            "b_k": jnp.zeros((K,)),
            "b_s_k": jnp.zeros((S, K)),
            "e_s_att": noise(keys[7], (S, config.d_att)),
            "v_i_att": noise(keys[8], (I, config.d_att)),
            "k_k_att": noise(keys[9], (K, config.d_att)),
        },
        "affect": {
            "a_s": noise(keys[10], (S, config.d_aff)),
            "mlp_w1": noise(keys[11], (config.d_aff + K, Ha)),
            "mlp_b1": jnp.zeros((Ha,)),
            "mlp_w2": noise(keys[12], (Ha, config.d_aff)),
            "mlp_b2": jnp.zeros((config.d_aff,)),
            "w_g": noise(keys[13], (config.d_aff,)),
            "b_g": jnp.zeros(()),
            "w_slip": jnp.zeros((config.d_aff,)),
            "b_slip": jnp.array(-4.0),
        },
        "gamma_si": jnp.asarray(config.gamma_si),
        "ln_eps": jnp.asarray(config.ln_eps),
        "prob_eps": jnp.asarray(config.prob_eps),
    }
    # Guess/slip heads start near (g, s) ≈ (0.5, small) via b_slip=-4 and
    # small w; residual b terms are exactly 0 as required by eq. 11–14 init.
    return params


def _gather(table, ids):
    return table[ids]


def forward_from_features(
    e_s,
    d_disc,
    d_kc,
    q_i,
    ncdm_mlp,
    b_s,
    b_i,
    u_s,
    v_i,
    gamma_si,
    w_si,
    w_c,
    b_k,
    b_s_k,
    e_s_att,
    v_i_att,
    k_k_att,
    a_s,
    affect_w,
    ln_eps=LN_EPS,
    prob_eps=PROB_EPS,
):
    """ARACD map from already-gathered feature tensors.

    Leading batch axes are optional; trailing concept/feature axes are required.
    ncdm_mlp is (w1, b1, w2, b2). affect_w is a dict of affect MLP/head weights.
    """
    y_base, x_si = ncdm_base(e_s, d_disc, d_kc, q_i, *ncdm_mlp)
    r_si = stu_item_residual(b_s, b_i, u_s, v_i, gamma_si=gamma_si)
    q_att = attention_query(e_s_att, v_i_att, eps=ln_eps)
    scores = attention_scores(q_att, k_k_att)
    alpha = q_masked_softmax(scores, q_i)
    r_c = concept_residual(alpha, b_k, b_s_k)
    r_ab = ability_residual(r_si, r_c, w_si, w_c)
    y_star = ability_correct(y_base, r_ab, eps=prob_eps)
    a_si = affect_mlp(
        a_s,
        d_kc,
        affect_w["mlp_w1"],
        affect_w["mlp_b1"],
        affect_w["mlp_w2"],
        affect_w["mlp_b2"],
    )
    g, s = guess_slip(a_si, affect_w["w_g"], affect_w["b_g"], affect_w["w_slip"], affect_w["b_slip"])
    y_hat = modulate(y_star, g, s)
    return {
        "y_hat": y_hat,
        "y_star": y_star,
        "y_base": y_base,
        "x_si": x_si,
        "r_stu_item": r_si,
        "r_concept": r_c,
        "r_ability": r_ab,
        "alpha": alpha,
        "scores": scores,
        "q_att": q_att,
        "a_si": a_si,
        "g": g,
        "s": s,
    }


def forward(params, student_ids, item_ids, q_matrix):
    """Batched ARACD forward for integer student/item ids.

    student_ids, item_ids: (B,) int
    q_matrix: (I, K)
    returns a dict of tensors with leading batch axis B.
    """
    student_ids = jnp.asarray(student_ids)
    item_ids = jnp.asarray(item_ids)
    ncdm = params["ncdm"]
    res = params["residual"]
    aff = params["affect"]

    e_s = sigmoid(_gather(ncdm["student_mastery"], student_ids))
    d_kc = sigmoid(_gather(ncdm["item_diff"], item_ids))
    d_disc = sigmoid(_gather(ncdm["item_disc"], item_ids))
    q_i = _gather(q_matrix, item_ids)

    ncdm_mlp = (ncdm["mlp_w1"], ncdm["mlp_b1"], ncdm["mlp_w2"], ncdm["mlp_b2"])
    return forward_from_features(
        e_s,
        d_disc,
        d_kc,
        q_i,
        ncdm_mlp,
        _gather(res["b_s"], student_ids),
        _gather(res["b_i"], item_ids),
        _gather(res["u_s"], student_ids),
        _gather(res["v_i"], item_ids),
        params["gamma_si"],
        res["w_si"],
        res["w_c"],
        res["b_k"],
        _gather(res["b_s_k"], student_ids),
        _gather(res["e_s_att"], student_ids),
        _gather(res["v_i_att"], item_ids),
        res["k_k_att"],
        _gather(aff["a_s"], student_ids),
        aff,
        ln_eps=params["ln_eps"],
        prob_eps=params["prob_eps"],
    )
