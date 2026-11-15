"""Dummy frozen full-vocabulary scorer. Not the paper's released backbone."""

from __future__ import annotations

import numpy as np


def frequency_scores(events, actor, relation, t, n_entities, floor=0.1):
    """p0(c|q) ∝ floor + count of (actor, relation, c) with time < t."""
    events = np.asarray(events)
    counts = np.full((n_entities,), floor, dtype=np.float64)
    if events.size:
        subj, rel, obj, time = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        hit = (subj == actor) & (rel == relation) & (time < t)
        for c in obj[hit]:
            c = int(c)
            if 0 <= c < n_entities:
                counts[c] += 1.0
    return counts / counts.sum()


def uniform_scores(n_entities):
    return np.full((n_entities,), 1.0 / n_entities, dtype=np.float64)


def random_scores(actor, relation, t, n_entities, seed=0):
    """Frozen dummy p0(c|q): softmax of a seeded Gaussian, independent of history."""
    rng = np.random.default_rng(seed + 1009 * int(actor) + 17 * int(relation) + int(t))
    z = rng.normal(size=n_entities)
    z = np.exp(z - z.max())
    return z / z.sum()
