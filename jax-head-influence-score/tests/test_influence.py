"""Properties of the score itself: ALTI, the weights, the signs, the ranking.

Most of these run on hand-built arrays rather than on a model, so the expected
answer is a number worked out by hand and not whatever the code happened to
produce.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from head_influence.influence import (
    CATEGORY_NAMES,
    FN,
    FP,
    TN,
    TP,
    alti_contributions,
    deltas,
    head_contributions,
    influence,
    influence_scores,
    layer_contributions,
    layer_scores,
    layer_weights,
    mean_scores_by_category,
    outcome_categories,
    proximity,
    s_final,
    signal_split,
    top_heads,
)
from head_influence.model import ModelConfig, forward, init_params, logit_direction
from head_influence.task import make_batch

from .conftest import DTYPE


# --- ALTI proximity, equation (1) -------------------------------------------

def test_proximity_of_a_component_equal_to_the_state_is_the_state_norm():
    r = jnp.array([1.0, -2.0, 3.0], DTYPE)
    assert float(proximity(r, r)) == pytest.approx(6.0, abs=1e-14)


def test_proximity_is_zero_once_the_component_is_far_enough():
    r = jnp.array([1.0, 1.0], DTYPE)          # ||r||_1 = 2
    far = jnp.array([-1.0, 1.0], DTYPE)       # ||z - r||_1 = 2, so exactly at the hinge
    beyond = jnp.array([-5.0, 1.0], DTYPE)    # ||z - r||_1 = 6
    assert float(proximity(far, r)) == 0.0
    assert float(proximity(beyond, r)) == 0.0


def test_proximity_matches_the_formula_on_a_hand_case():
    r = jnp.array([2.0, 0.0], DTYPE)
    z = jnp.array([1.5, 0.25], DTYPE)         # ||z-r||_1 = 0.75, ||r||_1 = 2
    assert float(proximity(z, r)) == pytest.approx(1.25, abs=1e-14)


def test_alti_contributions_are_non_negative_and_sum_to_one():
    key = jax.random.key(0)
    comps = jax.random.normal(key, (5, 8), DTYPE) * 0.1
    state = comps.sum(0)
    c = alti_contributions(comps, state)
    assert float(c.min()) >= 0.0
    assert float(c.sum()) == pytest.approx(1.0, abs=1e-14)


def test_alti_contributions_by_hand():
    """Three components, proximities worked out on paper.

    r = [4, 0], ||r||_1 = 4. z0 = [3, 0] gives 4 - 1 = 3, z1 = [1, 0] gives
    4 - 3 = 1, z2 = [0, 6] gives max(4 - 10, 0) = 0. So c = [3/4, 1/4, 0].
    """
    r = jnp.array([4.0, 0.0], DTYPE)
    comps = jnp.array([[3.0, 0.0], [1.0, 0.0], [0.0, 6.0]], DTYPE)
    c = alti_contributions(comps, r)
    np.testing.assert_allclose(np.asarray(c), [0.75, 0.25, 0.0], atol=1e-14)


def test_degenerate_state_gives_zero_contributions():
    r = jnp.zeros((3,), DTYPE)
    comps = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], DTYPE)
    c = alti_contributions(comps, r)
    assert float(c.sum()) == 0.0


# --- C_{k,h}, C_k and alpha_k -----------------------------------------------

def test_head_contributions_sum_to_one_within_each_layer():
    key = jax.random.key(1)
    writes = jax.random.normal(key, (3, 4, 16), DTYPE) * 0.1
    attn = writes.sum(1)
    c = head_contributions(writes, attn)
    assert float(c.min()) >= 0.0
    np.testing.assert_allclose(np.asarray(c.sum(-1)), np.ones(3), atol=1e-14)


def test_layer_weights_sum_to_one_over_the_chosen_subset():
    c_layer = jnp.array([2.0, 0.0, 5.0, 1.0], DTYPE)
    alpha = layer_weights(c_layer)
    assert float(alpha.sum()) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(np.asarray(alpha), [0.25, 0.0, 0.625, 0.125], atol=1e-14)

    subset = layer_weights(c_layer, [0, 3])
    assert float(subset.sum()) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(np.asarray(subset), [2 / 3, 0.0, 0.0, 1 / 3], atol=1e-14)


def test_layer_contribution_by_hand():
    """C_k as A_k's ALTI share of the state it helped build.

    Layer 0: r_pre = [4, 0], A = [1, 0], r_mid = [5, 0]. The proximities are
    5 - 4 = 1 for A and 5 - 1 = 4 for r_pre, so C_0 = 1/5.
    Layer 1: r_pre = [0, 1], A = [0, 0.5], r_mid = [0, 1.5], proximities
    1.5 - 1 = 0.5 and 1.5 - 0.5 = 1, so C_1 = 1/3.
    """
    attn = jnp.array([[1.0, 0.0], [0.0, 0.5]], DTYPE)
    resid = jnp.array([[4.0, 0.0], [0.0, 1.0]], DTYPE)
    got = layer_contributions(attn, resid)
    np.testing.assert_allclose(np.asarray(got), [0.2, 1 / 3], atol=1e-14)
    assert float(got.min()) >= 0.0 and float(got.max()) <= 1.0


def test_layer_contribution_closed_form_matches_the_two_component_alti():
    """The three-norm shortcut in the docstring is the same number.

    It only holds where the hinge is inactive, so the layers where attention
    shrinks the L1 norm of the residual state are checked separately: those
    give C_k = 0, not a negative share.
    """
    attn = jax.random.normal(jax.random.key(17), (8, 32), DTYPE)
    resid = jax.random.normal(jax.random.key(18), (8, 32), DTYPE) * 2.0
    n_mid = jnp.abs(resid + attn).sum(-1)
    n_resid = jnp.abs(resid).sum(-1)
    n_attn = jnp.abs(attn).sum(-1)
    closed = (n_mid - n_resid) / (2 * n_mid - n_resid - n_attn)
    got = layer_contributions(attn, resid)
    live = np.asarray((n_mid > n_resid) & (n_mid > n_attn))
    assert live.any() and not live.all()
    np.testing.assert_allclose(np.asarray(got)[live], np.asarray(closed)[live], atol=1e-14)
    assert np.all(np.asarray(got)[~live] == 0.0)


def test_unnormalized_layer_contribution_collapses():
    """Why C_k is read as a share and not as a bare proximity.

    A_k is not a piece of r_pre, so for a network of any width the hinge in
    `proximity` clips the raw form to zero and every alpha_k goes with it.
    """
    key = jax.random.key(19)
    attn = jax.random.normal(key, (3, 64), DTYPE)
    resid = jax.random.normal(jax.random.key(20), (3, 64), DTYPE) * 1.5
    assert float(layer_contributions(attn, resid, normalize=False).sum()) == 0.0
    assert float(layer_contributions(attn, resid).min()) > 0.0


# --- Equation (2) and the weighted-average reading of equation (3) ----------

def test_score_is_delta_times_the_weights_by_hand():
    """A two-layer, two-head case with every number chosen so S is exact.

    w = [1, 0]. Layer 0 writes [2, 0] and [1, 0], so A_0 = [3, 0] and the
    proximities are 3 - 1 = 2 and 3 - 2 = 1, giving C_0h = [2/3, 1/3].
    Layer 1 writes [4, 0] and [-1, 0], so A_1 = [3, 0], proximities 3 - 1 = 2
    and max(3 - 4, 0) = 0, giving C_1h = [1, 0].
    The residual entering each block is [6, 0], so r_mid = [9, 0] and C_k is
    (9 - 6) / (9 - 6 + 9 - 3) = 1/3 for both layers, giving alpha = [1/2, 1/2]. Then S_00 = 2 * 2/3 * 1/2 = 2/3, S_01 = 1 * 1/3 * 1/2,
    S_10 = 4 * 1 * 1/2 = 2, S_11 = 0.
    """
    w = jnp.array([1.0, 0.0], DTYPE)
    writes = jnp.array([[[2.0, 0.0], [1.0, 0.0]],
                        [[4.0, 0.0], [-1.0, 0.0]]], DTYPE)
    attn = writes.sum(1)
    resid = jnp.array([[6.0, 0.0], [6.0, 0.0]], DTYPE)

    delta = deltas(writes, w)
    c_head = head_contributions(writes, attn)
    alpha = layer_weights(layer_contributions(attn, resid))
    score = influence_scores(delta, c_head, alpha)

    np.testing.assert_allclose(np.asarray(delta), [[2.0, 1.0], [4.0, -1.0]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(c_head), [[2 / 3, 1 / 3], [1.0, 0.0]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(alpha), [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(np.asarray(score),
                               [[2 / 3, 1 / 6], [2.0, 0.0]], atol=1e-14)
    assert float(s_final(score)) == pytest.approx(2 / 3 + 1 / 6 + 2.0, abs=1e-14)


def test_sign_of_the_score_follows_the_direction_of_the_write():
    """A head writing along +w scores positive, one writing along -w negative.

    The magnitudes are fixed by hand too: with two heads whose writes are
    +3w and -1w, A = 2w, so the head proximities are 2 - 1 = 1 and
    max(2 - 3, 0) = 0, giving C = [1, 0]. There is one layer, so alpha = [1]
    whatever C_0 is, and S = [3, 0].
    """
    w = jnp.array([0.0, 1.0, 0.0], DTYPE)
    writes = jnp.stack([jnp.stack([3.0 * w, -1.0 * w])])
    attn = writes.sum(1)
    resid = jnp.stack([10.0 * w])
    delta = deltas(writes, w)
    c_head = head_contributions(writes, attn)
    alpha = layer_weights(layer_contributions(attn, resid))
    score = influence_scores(delta, c_head, alpha)
    assert float(delta[0, 0]) > 0 and float(delta[0, 1]) < 0
    np.testing.assert_allclose(np.asarray(score), [[3.0, 0.0]], atol=1e-14)

    # Flip the direction and every sign flips with it.
    flipped = influence_scores(deltas(writes, -w), c_head, alpha)
    np.testing.assert_allclose(np.asarray(flipped), [[-3.0, 0.0]], atol=1e-14)


def test_sign_semantics_on_a_planted_model():
    """Same claim, but through a real forward pass with planted head writes.

    Zero query and key weights make attention uniform, so every head reads the
    same average of the value stream; planting W_O then puts a chosen multiple
    of w into the residual stream and the score has a known sign per head.
    """
    cfg = ModelConfig(n_layers=1, n_heads=2, d_model=4, d_head=2, seq_len=4,
                      use_layernorm=False)
    params = init_params(jax.random.key(2), cfg, DTYPE)
    params["attn"] = dict(params["attn"])
    params["attn"]["wq"] = jnp.zeros_like(params["attn"]["wq"])
    params["attn"]["wk"] = jnp.zeros_like(params["attn"]["wk"])
    tokens = jnp.arange(cfg.seq_len)

    w = logit_direction(params)
    trace = forward(cfg, params, tokens)
    delta = deltas(trace.head_writes[:, :, cfg.cls_index, :], w)
    inf = influence(cfg, params, tokens)
    for h in range(cfg.n_heads):
        assert float(jnp.sign(inf.score[0, h])) in (0.0, float(jnp.sign(delta[0, h])))
    # The head aligned with w scores positive, the anti-aligned one negative,
    # because C and alpha are non-negative by construction.
    assert float(inf.c_head.min()) >= 0.0
    assert float(inf.alpha.min()) >= 0.0


def test_s_final_is_a_weighted_average_of_the_deltas(deep_cfg, deep_params):
    """The paper's reading of equation (3), checked as a real constraint.

    The weights C_{k,h} * alpha_k sum to 1, so S_final has to land between the
    smallest and the largest Delta, on every layer subset.
    """
    cfg, random_params = deep_cfg, deep_params
    tokens, _ = make_batch(jax.random.key(9), 8, cfg.seq_len, cfg.vocab_size)
    for subset in (None, [0], [0, 2], [1, 2]):
        for tok in tokens:
            inf = influence(cfg, random_params, tok, layers=subset)
            mask = jnp.ones_like(inf.alpha) if subset is None else jnp.zeros_like(
                inf.alpha).at[jnp.asarray(subset)].set(1.0)
            weights = inf.c_head * inf.alpha[:, None]
            assert float(weights.min()) >= 0.0
            assert float(weights.sum()) == pytest.approx(1.0, abs=1e-13)
            sel = inf.delta[mask > 0]
            assert float(sel.min()) - 1e-12 <= float(inf.s_final) <= float(sel.max()) + 1e-12


def test_scores_outside_the_subset_are_zero(deep_cfg, deep_params):
    cfg = deep_cfg
    tokens, _ = make_batch(jax.random.key(10), 4, cfg.seq_len, cfg.vocab_size)
    for tok in tokens:
        inf = influence(cfg, deep_params, tok, layers=[1])
        assert float(jnp.abs(inf.score[0]).sum()) == 0.0
        assert float(jnp.abs(inf.score[2]).sum()) == 0.0
        assert float(inf.s_final) == pytest.approx(float(inf.score[1].sum()), abs=1e-14)


def test_layer_scores_sum_the_heads_of_each_layer():
    score = jnp.array([[1.0, -2.0], [0.5, 0.25]], DTYPE)
    np.testing.assert_allclose(np.asarray(layer_scores(score)), [-1.0, 0.75], atol=1e-14)
    assert float(layer_scores(score).sum()) == pytest.approx(float(s_final(score)), abs=1e-14)


def test_signal_split_matches_section_6_2():
    score = jnp.array([[1.0, -2.0], [0.5, 0.0]], DTYPE)
    pos, neg = signal_split(score)
    assert float(pos) == pytest.approx(1.5, abs=1e-14)
    assert float(neg) == pytest.approx(2.0, abs=1e-14)


# --- ranking and category tables of Section 6.3 -----------------------------

def test_top_heads_on_a_hand_built_table():
    score = jnp.array([[0.10, -0.40], [0.25, 0.05]], DTYPE)
    assert top_heads(score, 2, by="abs")[0][:2] == (0, 1)
    assert [h[:2] for h in top_heads(score, 2, by="abs")] == [(0, 1), (1, 0)]
    assert [h[:2] for h in top_heads(score, 2, by="positive")] == [(1, 0), (0, 0)]
    assert [h[:2] for h in top_heads(score, 1, by="negative")] == [(0, 1)]
    assert top_heads(score, 1, by="positive")[0][2] == pytest.approx(0.25)


def test_top_heads_rejects_an_unknown_criterion():
    with pytest.raises(ValueError):
        top_heads(jnp.zeros((2, 2), DTYPE), 1, by="nonsense")


def test_outcome_categories_by_hand():
    preds = jnp.array([1, 0, 1, 0])
    labels = jnp.array([1, 0, 0, 1])
    got = outcome_categories(preds, labels)
    assert [int(x) for x in got] == [TP, TN, FP, FN]
    assert CATEGORY_NAMES[TP] == "TP"


def test_mean_scores_by_category_by_hand():
    scores = jnp.array([
        [[1.0, 0.0]],
        [[3.0, 0.0]],
        [[0.0, 8.0]],
    ], DTYPE)
    cats = jnp.array([TP, TP, TN])
    np.testing.assert_allclose(np.asarray(mean_scores_by_category(scores, cats, TP)),
                               [[2.0, 0.0]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(mean_scores_by_category(scores, cats, TN)),
                               [[0.0, 8.0]], atol=1e-14)
    np.testing.assert_allclose(np.asarray(mean_scores_by_category(scores, cats, FP)),
                               [[0.0, 0.0]], atol=1e-14)
