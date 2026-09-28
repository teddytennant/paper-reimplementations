"""Filtered ranking on a toy TKG. Not paper table numbers."""

from __future__ import annotations

import numpy as np


def filtered_ranks(scores, gold, filter_sets):
    """1-based rank of gold after removing other true objects for this query."""
    ranks = []
    for i, g in enumerate(gold):
        s = np.array(scores[i], dtype=np.float64, copy=True)
        for o in filter_sets[i]:
            if int(o) != int(g):
                s[int(o)] = -np.inf
        ranks.append(1 + int(np.sum(s > s[int(g)])))
    return np.asarray(ranks, dtype=np.int32)


def filtered_mrr(scores, gold, filter_sets):
    ranks = filtered_ranks(scores, gold, filter_sets)
    return float(np.mean(1.0 / ranks))


def hits_at_k(scores, gold, filter_sets, k=1):
    ranks = filtered_ranks(scores, gold, filter_sets)
    return float(np.mean(ranks <= k))
