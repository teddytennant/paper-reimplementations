"""Damped accumulation of structure-map means along quadtree ancestors.

For a token at depth ``d`` (root depth 0):

    sum_{j=0}^{d} mean(S over the ancestor rectangle at depth j) / sqrt(j + 1)

Ancestors are the aligned dyadic parents of ``token_rect`` implied by equal
quadrant splits of ``structure``. The added term at depth ``j`` is the local
mean divided by ``sqrt(j + 1)``, which grows sublinearly in depth.
"""

from __future__ import annotations

import math

import numpy as np


def increment(j, local_mean=1.0):
    """Added term at depth ``j``: ``local_mean / sqrt(j + 1)``."""
    return float(local_mean) / math.sqrt(j + 1)


def _parent_rect(rect):
    r0, r1, c0, c1 = rect
    height = r1 - r0
    width = c1 - c0
    parent_h = height * 2
    parent_w = width * 2
    pr0 = (r0 // parent_h) * parent_h
    pc0 = (c0 // parent_w) * parent_w
    return (pr0, pr0 + parent_h, pc0, pc0 + parent_w)


def damped_variance(structure, token_rect, depth):
    """Accumulated damped variance of one token. Returns a Python float."""
    structure = np.asarray(structure, dtype=np.float64)
    rect = tuple(int(x) for x in token_rect)
    depth = int(depth)
    total = 0.0
    for j in range(depth, -1, -1):
        r0, r1, c0, c1 = rect
        total += float(structure[r0:r1, c0:c1].mean()) / math.sqrt(j + 1)
        if j == 0:
            break
        rect = _parent_rect(rect)
    return float(total)


def canvas(structure, tokens, depths):
    """Per-token damped variances, one entry per token, in list order."""
    values = [
        damped_variance(structure, token, depth)
        for token, depth in zip(tokens, depths)
    ]
    return np.asarray(values, dtype=np.float64)
