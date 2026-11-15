"""Scoring, attention, training, reciprocal params, and filtered ranks."""

import jax
import jax.numpy as jnp
import numpy as np

from bsca import filtered_ranks, fuse_attention, init_params, score, train_step


def _toy_facts():
    facts = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [1, 0, 2, 0],
            [1, 0, 2, 1],
            [2, 1, 3, 0],
            [2, 1, 3, 2],
            [3, 1, 4, 1],
            [4, 2, 5, 0],
            [4, 2, 5, 3],
            [5, 2, 0, 2],
            [5, 2, 1, 3],
        ],
        dtype=np.int32,
    )
    return facts


def test_score_is_finite_real_scalar():
    params = init_params(jax.random.PRNGKey(0), 6, 3, 4, k=4)
    s = score(params, 0, 1, 2, 3)
    assert s.shape == ()
    assert jnp.issubdtype(s.dtype, jnp.floating)
    assert np.isfinite(np.asarray(s))


def test_attention_weights_are_real_simplex():
    params = init_params(jax.random.PRNGKey(1), 6, 3, 4, k=4)
    e_s = params["entities"][0]
    e_tau, e_rel = e_s + params["time_add"][1], e_s + params["rel_add"][0]
    fused, (a_tau, a_rel) = fuse_attention(params["attn"], e_tau, e_rel)
    assert a_tau.shape == ()
    assert a_rel.shape == ()
    assert jnp.issubdtype(a_tau.dtype, jnp.floating)
    assert jnp.issubdtype(a_rel.dtype, jnp.floating)
    assert float(a_tau) >= 0.0 and float(a_rel) >= 0.0
    np.testing.assert_allclose(float(a_tau + a_rel), 1.0, atol=1e-6)
    assert fused.shape == e_s.shape


def test_additive_zero_and_unit_scale_leave_score():
    params = init_params(jax.random.PRNGKey(2), 6, 3, 4, k=4)
    s0 = score(params, 1, 0, 2, 1)
    shifted = dict(params)
    shifted["entities"] = params["entities"] + jnp.zeros_like(params["entities"])
    np.testing.assert_allclose(np.asarray(score(shifted, 1, 0, 2, 1)), np.asarray(s0), atol=1e-6)
    scaled = dict(params)
    scaled["entities"] = params["entities"] * jnp.array(1.0, dtype=params["entities"].real.dtype)
    np.testing.assert_allclose(np.asarray(score(scaled, 1, 0, 2, 1)), np.asarray(s0), atol=1e-6)
    translated = dict(params)
    delta = jnp.ones((1, 4, params["entities"].shape[-1]), dtype=params["entities"].dtype)
    translated["entities"] = params["entities"] + delta
    s_tr = score(translated, 1, 0, 2, 1)
    assert not np.isclose(np.asarray(s_tr), np.asarray(s0), atol=1e-5)


def test_tiny_tkg_true_scores_beat_corrupted():
    n_ent, n_rel, n_tau, k = 6, 3, 4, 4
    facts = _toy_facts()
    params = init_params(jax.random.PRNGKey(7), n_ent, n_rel, n_tau, k=k, scale=0.2)
    batch = (facts[:, 0], facts[:, 1], facts[:, 2], facts[:, 3])
    state = None
    for _ in range(80):
        params, state, _loss = train_step(
            params, batch, state, lr=0.1, lambda_n3=1e-4, lambda_tau=1e-4
        )
    true_scores = np.array([float(score(params, int(s), int(r), int(o), int(t))) for s, r, o, t in facts])
    rng = np.random.default_rng(11)
    true_set = {tuple(row.tolist()) for row in facts}
    corrupted = []
    for s, r, o, t in facts:
        for _try in range(20):
            o2 = int(rng.integers(0, n_ent))
            if o2 != int(o) and (int(s), int(r), o2, int(t)) not in true_set:
                corrupted.append((int(s), int(r), o2, int(t)))
                break
    assert len(corrupted) == len(facts)
    corr_scores = np.array([float(score(params, s, r, o, t)) for s, r, o, t in corrupted])
    assert true_scores.mean() > corr_scores.mean()


def test_reciprocal_params_and_subject_loss_finite():
    from bsca.loss import multiclass_loss

    n_rel = 3
    params = init_params(jax.random.PRNGKey(3), 6, n_rel, 4, k=4)
    assert params["rel_mul"].shape[0] == 2 * n_rel
    assert params["rel_add"].shape[0] == 2 * n_rel
    facts = _toy_facts()
    loss = multiclass_loss(
        params, facts[:, 0], facts[:, 1], facts[:, 2], facts[:, 3]
    )
    assert np.isfinite(float(loss))
    inv_score = score(params, int(facts[0, 2]), int(facts[0, 1]) + n_rel, int(facts[0, 0]), int(facts[0, 3]))
    assert np.isfinite(float(inv_score))


def test_filtered_rank_at_most_n_entities():
    n_ent, n_rel, n_tau, k = 6, 3, 4, 4
    facts = _toy_facts()
    params = init_params(jax.random.PRNGKey(4), n_ent, n_rel, n_tau, k=k)
    s, r, o, tau = facts[0]
    rank = filtered_ranks(params, s, r, o, tau, facts=facts)
    assert int(rank) >= 1
    assert int(rank) <= n_ent
    ranks = filtered_ranks(params, facts[:, 0], facts[:, 1], facts[:, 2], facts[:, 3], facts=facts)
    assert ranks.shape == (len(facts),)
    assert int(np.max(np.asarray(ranks))) <= n_ent
    assert int(np.min(np.asarray(ranks))) >= 1
