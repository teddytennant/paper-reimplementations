"""The toy task, end to end: does the head actually learn to rank."""

import numpy as np
import pytest

from hllm.data import permutation_matrices, sample_slates
from hllm.hungarian import decode_ranking
from hllm.metrics import chance_ndcg, exact_match, ndcg, top1_accuracy
from hllm.train import TrainConfig, train

import jax


def small(**kw):
    base = dict(steps=200, batch=32, n_items=6, d_model=16, width=32, n_heads=4,
                eval_slates=128, log_every=50, seed=0)
    base.update(kw)
    return TrainConfig(**base)


def test_teacher_permutation_matrices_round_trip():
    """P_teacher and the decoder have to agree on what a ranking is: feeding a
    permutation matrix to the decoder returns the ranking it was built from."""
    slates = sample_slates(jax.random.key(0), 16, 7, 8)
    teacher = np.asarray(slates.teacher)
    order = np.asarray(slates.order)
    for i in range(16):
        assert teacher[i].sum(0).tolist() == [1] * 7
        assert teacher[i].sum(1).tolist() == [1] * 7
        assert np.array_equal(decode_ranking(teacher[i]), order[i])


def test_teacher_ranking_sorts_by_relevance():
    slates = sample_slates(jax.random.key(1), 8, 9, 8, noise=0.0)
    rel = np.asarray(slates.relevance)
    order = np.asarray(slates.order)
    got = np.take_along_axis(rel, order, axis=-1)
    assert (np.diff(got, axis=-1) <= 0).all()


def test_metrics_are_sane_at_the_extremes():
    order = np.array([[3, 0, 2, 1], [1, 2, 3, 0]])
    assert ndcg(order, order).tolist() == pytest.approx([1.0, 1.0])
    assert exact_match(order, order).tolist() == [1.0, 1.0]
    assert top1_accuracy(order, order).tolist() == [1.0, 1.0]
    reversed_order = order[:, ::-1]
    assert (ndcg(reversed_order, order) < 0.8).all()
    assert exact_match(reversed_order, order).tolist() == [0.0, 0.0]
    swapped = order.copy()
    swapped[:, [2, 3]] = swapped[:, [3, 2]]
    # A swap at the tail costs less than a swap at the head.
    head_swap = order.copy()
    head_swap[:, [0, 1]] = head_swap[:, [1, 0]]
    assert (ndcg(swapped, order) > ndcg(head_swap, order)).all()


def test_permutation_matrices_accept_more_positions_than_items():
    order = np.array([[2, 0, 1]])
    p = np.asarray(permutation_matrices(order, 5))
    assert p.shape == (1, 3, 5)
    assert p[0].sum() == 3


@pytest.mark.parametrize("head", ["linear_probe", "self_attention", "slot_query"])
def test_every_head_learns_to_rank(head):
    """NDCG on held-out slates has to clear a random ranking by a wide margin,
    and the exact permutation has to come out far more often than 1 / N!."""
    _, r = train(small(head=head), verbose=False)
    assert r["chance_ndcg"] == pytest.approx(0.79, abs=0.05)
    assert r["ndcg"] > r["chance_ndcg"] + 0.15
    assert r["ndcg"] > 0.97
    assert r["recall@1"] > 0.75
    assert r["exact"] > 0.2  # chance is 1 / 720
    assert r["loss"] < 1.0


def test_training_is_deterministic():
    _, a = train(small(steps=60), verbose=False)
    _, b = train(small(steps=60), verbose=False)
    assert a["ndcg"] == b["ndcg"]
    assert a["loss"] == b["loss"]


def test_untrained_head_is_no_better_than_chance():
    """The margin above has to come from training, not from the metric."""
    _, r = train(small(steps=0), verbose=False)
    assert abs(r["ndcg"] - r["chance_ndcg"]) < 0.12


def test_decoded_rankings_are_valid_permutations_after_training():
    from hllm.heads import HeadConfig, apply_head, init_head

    cfg = small(steps=40)
    params, _ = train(cfg, verbose=False)
    head_cfg = HeadConfig(kind=cfg.head, d_model=cfg.d_model, n_positions=cfg.n_items,
                          width=cfg.width, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    slates = sample_slates(jax.random.key(99), 64, cfg.n_items, cfg.d_model)
    scores = jax.vmap(lambda h: apply_head(head_cfg, params, h))(slates.hidden)
    for m in np.asarray(scores):
        order = decode_ranking(m)
        assert np.array_equal(np.sort(order), np.arange(cfg.n_items))


def test_chance_baseline_is_stable():
    slates = sample_slates(jax.random.key(3), 512, 8, 8)
    order = np.asarray(slates.order)
    values = [
        float(chance_ndcg(np.random.default_rng(s), order).mean()) for s in range(5)
    ]
    assert max(values) - min(values) < 0.03
    assert 0.7 < np.mean(values) < 0.85
