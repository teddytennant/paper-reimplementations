"""Causal validation, Section 6.4.

Three of these run on models built by hand, where the head that carries the
decision is known before the score is computed. The last two run on the trained
classifier and compare the heads the score picks against every other choice of
the same size, so the comparison is against a whole distribution rather than one
random draw that would flip sign between machines.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from head_influence.ablation import (
    accuracy,
    accuracy_for_masks,
    all_subset_masks,
    head_mask,
)
from head_influence.influence import batched_influence, influence, top_heads
from head_influence.model import ModelConfig, batched_logits, init_params
from head_influence.task import CLS, TRIGGER, make_batch

from .conftest import DTYPE

# A planted model: 4 positions, 4 token types, and only the dimensions the
# construction uses are non-zero.
PLANTED = ModelConfig(vocab_size=4, seq_len=4, n_layers=2, n_heads=2, d_model=8,
                      d_head=4, d_mlp=8, use_layernorm=False, cls_index=0)
BETA = 2.0     # readout scale
GAIN = 3.0     # head write scale


def _zeroed(cfg):
    params = init_params(jax.random.key(0), cfg, DTYPE)
    zeros = lambda tree: jax.tree.map(jnp.zeros_like, tree)
    params = dict(params)
    params["attn"] = zeros(params["attn"])
    params["mlp"] = zeros(params["mlp"])
    params["tok_embed"] = jnp.zeros_like(params["tok_embed"])
    params["pos_embed"] = jnp.zeros_like(params["pos_embed"])
    params["readout"] = dict(params["readout"])
    params["readout"]["w"] = jnp.zeros_like(params["readout"]["w"])
    params["readout"]["b"] = jnp.zeros_like(params["readout"]["b"])
    return params


def _planted_reporter():
    """One head reads the trigger flag straight into the logit direction.

    Feature 0 of the embedding flags the trigger token. Query and key weights
    are zero, so attention is uniform and head (1, 0) sees the fraction of
    trigger tokens in the sequence; W_O turns that into GAIN * frac along
    feature 1, which the readout reads as the class difference. Every other
    head writes nothing. The threshold sits below one trigger in four
    positions, so the model answers "is there a trigger" exactly.
    """
    cfg = PLANTED
    p = _zeroed(cfg)
    p["tok_embed"] = p["tok_embed"].at[TRIGGER, 0].set(1.0)
    p["attn"]["wv"] = p["attn"]["wv"].at[1, 0, 0, 0].set(1.0)
    p["attn"]["wo"] = p["attn"]["wo"].at[1, 0, 0, 1].set(GAIN)
    p["readout"]["w"] = p["readout"]["w"].at[1, 1].set(BETA).at[0, 1].set(-BETA)
    # frac is 1/4 for one trigger in four positions; put the threshold halfway.
    p["readout"]["b"] = p["readout"]["b"].at[1].set(-0.5 * BETA * GAIN * 0.25)
    return cfg, p


def _planted_composition():
    """The decision needs a head whose own write points nowhere near w.

    Head (0, 0) moves the trigger flag from feature 0 to feature 2; head (1, 0)
    reads feature 2 and writes the answer along feature 1. Deleting (0, 0)
    destroys the model, but its write is orthogonal to the logit direction, so
    Delta_{0,0} and therefore S_{0,0} are exactly zero. Direct projection onto
    the logit cannot see a head that only matters through composition, and the
    paper's score inherits that.
    """
    cfg = PLANTED
    p = _zeroed(cfg)
    p["tok_embed"] = p["tok_embed"].at[TRIGGER, 0].set(1.0)
    p["attn"]["wv"] = p["attn"]["wv"].at[0, 0, 0, 0].set(1.0)
    p["attn"]["wo"] = p["attn"]["wo"].at[0, 0, 0, 2].set(1.0)
    p["attn"]["wv"] = p["attn"]["wv"].at[1, 0, 2, 0].set(1.0)
    p["attn"]["wo"] = p["attn"]["wo"].at[1, 0, 0, 1].set(GAIN)
    p["readout"]["w"] = p["readout"]["w"].at[1, 1].set(BETA).at[0, 1].set(-BETA)
    # Both hops average uniformly, so the flag still arrives as frac and the
    # threshold is the same one the reporter uses.
    p["readout"]["b"] = p["readout"]["b"].at[1].set(-0.5 * BETA * GAIN * 0.25)
    return cfg, p


def _planted_data(cfg):
    """Sequences with and without a trigger, half of each."""
    tokens = jnp.full((8, cfg.seq_len), 2, dtype=jnp.int32).at[:, 0].set(CLS)
    labels = jnp.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    where = jnp.array([1, 1, 2, 2, 3, 3, 1, 3])
    for i in range(8):
        if labels[i] == 1:
            tokens = tokens.at[i, where[i]].set(TRIGGER)
    return tokens, labels


def test_planted_reporter_is_the_top_scoring_head():
    cfg, params = _planted_reporter()
    tokens, labels = _planted_data(cfg)
    assert float(accuracy(cfg, params, tokens, labels)) == 1.0

    inf = batched_influence(cfg, params, tokens)
    mean_abs = jnp.abs(inf.score).mean(0)
    assert top_heads(mean_abs, 1)[0][:2] == (1, 0)
    # Only that head writes anything, so the contributions are exact:
    # C_{1,0} = 1, alpha = [0, 1], and S = Delta = BETA * GAIN * frac * 2.
    one = influence(cfg, params, tokens[1])
    np.testing.assert_allclose(np.asarray(one.alpha), [0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(np.asarray(one.c_head), [[0.0, 0.0], [1.0, 0.0]], atol=1e-14)
    expected = 2 * BETA * GAIN * (1 / cfg.seq_len)
    assert float(one.score[1, 0]) == pytest.approx(expected, abs=1e-13)
    assert float(one.s_final) == pytest.approx(expected, abs=1e-13)


def test_ablating_the_planted_head_destroys_the_model():
    cfg, params = _planted_reporter()
    tokens, labels = _planted_data(cfg)
    base = float(accuracy(cfg, params, tokens, labels))
    killed = float(accuracy(cfg, params, tokens, labels, head_mask(cfg, [(1, 0)], DTYPE)))
    assert base == 1.0 and killed == 0.5  # falls back to always answering "no"
    for layer, head in [(0, 0), (0, 1), (1, 1)]:
        spared = float(accuracy(cfg, params, tokens, labels,
                                head_mask(cfg, [(layer, head)], DTYPE)))
        assert spared == 1.0


def test_score_is_blind_to_a_head_that_only_matters_through_composition():
    cfg, params = _planted_composition()
    tokens, labels = _planted_data(cfg)
    assert float(accuracy(cfg, params, tokens, labels)) == 1.0

    inf = batched_influence(cfg, params, tokens)
    mean_abs = jnp.abs(inf.score).mean(0)
    assert float(mean_abs[0, 0]) == 0.0
    assert float(mean_abs[1, 0]) > 0.0
    # And yet removing it is exactly as fatal as removing the head that does
    # score, since the two sit in series.
    for layer, head in [(0, 0), (1, 0)]:
        killed = float(accuracy(cfg, params, tokens, labels,
                                head_mask(cfg, [(layer, head)], DTYPE)))
        assert killed == 0.5


def test_s_final_separates_the_two_decisions(trained):
    """The sign convention of Section 5, on weights that were actually fit.

    S_final is a weighted average of the Delta values and the paper reads its
    sign as the direction of the decision, so the examples this classifier
    calls positive have to sit above the ones it calls negative. Across eight
    training seeds the gap ran from 3.0 to 6.1.
    """
    cfg, params, acc = trained
    assert acc > 0.95
    tokens, labels = make_batch(jax.random.key(100), 256, cfg.seq_len, cfg.vocab_size)
    inf = batched_influence(cfg, params, tokens)
    preds = np.asarray(batched_logits(cfg, params, tokens, None).argmax(-1)).astype(bool)
    s_final = np.asarray(inf.s_final)
    assert preds.any() and (~preds).any()
    assert s_final[preds].mean() - s_final[~preds].mean() > 1.0


def test_dominant_heads_carry_more_of_the_decision_than_other_heads(trained):
    """Section 6.4, kept to the claim the score can support here.

    Every C(6, n) choice of n heads is enumerated and ranked by how much of the
    logit difference it accounts for, so the comparison runs against a whole
    distribution instead of one random draw that would flip sign on other
    hardware. Over eight training seeds the heads the score picks never fell
    below the 85th percentile of that distribution, and ranking by C_{k,h} *
    alpha_k alone, the paper's "contrib only" criterion, dropped as low as the
    13th.

    The paper's own version of this, rerunning the model without those heads
    and watching accuracy fall, does not survive here. Over the same eight
    seeds the top-|S| pair beat the mean over all fifteen pairs four times, and
    the top triple three times out of eight, which is a coin flip.
    `test_score_is_blind_to_a_head_that_only_matters_through_composition` has
    the reason: this task needs a first-layer head that writes nothing toward
    the logit, removing that head is what actually breaks the model, and a
    score built on a direct projection cannot rank it.
    """
    cfg, params, acc = trained
    tokens, labels = make_batch(jax.random.key(100), 256, cfg.seq_len, cfg.vocab_size)
    inf = batched_influence(cfg, params, tokens)
    delta = np.asarray(inf.delta)
    mean_abs = jnp.abs(inf.score).mean(0)
    base = float(accuracy(cfg, params, tokens, labels))

    for n in (2, 3):
        chosen = [(k, h) for k, h, _ in top_heads(mean_abs, n)]
        _, subsets = all_subset_masks(cfg, n)
        carried = np.array([
            np.abs(delta[:, [k for k, _ in sub], [h for _, h in sub]].sum(1)).mean()
            for sub in subsets])
        idx = [i for i, sub in enumerate(subsets) if set(sub) == set(chosen)][0]
        assert float((carried < carried[idx]).mean()) >= 0.7
        assert carried[idx] > carried.mean()

    # Rerunning the model without a pair of heads does cost accuracy on
    # average, even though which pair the score picks is not the worst one.
    masks, _ = all_subset_masks(cfg, 2)
    damage = base - np.asarray(accuracy_for_masks(cfg, params, tokens, labels, masks))
    assert damage.mean() > 0.02
