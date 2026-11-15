"""Ranking metrics for the toy task.

The paper reports AUC, Recall@{1, 10} and NDCG@1 against a real teacher. On the
synthetic task the teacher permutation is the only label, so relevance is taken
to be rank based: the item the teacher put first has gain N - 1, the next N - 2,
and so on. NDCG is then 1 exactly when the decoded ranking matches the teacher.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ndcg", "exact_match", "top1_accuracy", "chance_ndcg"]


def _gains_from_order(order):
    """Gain of every item, from the teacher's ranking."""
    order = np.asarray(order)
    n = order.shape[-1]
    gains = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(gains, order, np.arange(n - 1, -1, -1, dtype=np.float64), axis=-1)
    return gains


def ndcg(predicted_order, teacher_order, k: int | None = None) -> np.ndarray:
    """NDCG@k of a decoded ranking against the teacher's, per slate."""
    predicted_order = np.asarray(predicted_order)
    gains = _gains_from_order(teacher_order)
    n = predicted_order.shape[-1]
    k = n if k is None else min(k, n)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    got = np.take_along_axis(gains, predicted_order[..., :k], axis=-1)
    ideal = np.sort(gains, axis=-1)[..., ::-1][..., :k]
    return (got * discount).sum(-1) / (ideal * discount).sum(-1)


def exact_match(predicted_order, teacher_order) -> np.ndarray:
    """1.0 where the whole permutation is right, per slate."""
    predicted_order = np.asarray(predicted_order)
    teacher_order = np.asarray(teacher_order)
    return (predicted_order == teacher_order).all(-1).astype(np.float64)


def top1_accuracy(predicted_order, teacher_order) -> np.ndarray:
    """Recall@1: did rank 1 go to the item the teacher put first."""
    predicted_order = np.asarray(predicted_order)
    teacher_order = np.asarray(teacher_order)
    return (predicted_order[..., 0] == teacher_order[..., 0]).astype(np.float64)


def chance_ndcg(rng, teacher_order, k: int | None = None) -> np.ndarray:
    """NDCG of a uniformly random ranking, the baseline to beat."""
    teacher_order = np.asarray(teacher_order)
    shuffled = np.argsort(rng.random(teacher_order.shape), axis=-1)
    return ndcg(shuffled, teacher_order, k=k)
