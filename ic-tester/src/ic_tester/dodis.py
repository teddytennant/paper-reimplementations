"""Dodis pair distribution (Lemma 3.8 / Eq. (1)).

ν2(r) is the 2-adic valuation of a positive integer r. 𝒟 is the uniform
distribution over pairs (p, l) in [0:m]^2 satisfying

    0 < l - p <= 2^{max(ν2(p+1), ν2(l+1))}.

The bid domain is the integer interval {0, 1, ..., m}.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


def nu2(r: int) -> int:
    """Largest t such that 2^t divides the positive integer r."""
    if r <= 0:
        raise ValueError(f"nu2 is defined for positive integers, got r={r}")
    t = 0
    while r % 2 == 0:
        r //= 2
        t += 1
    return t


def _satisfies_eq1(p: int, l: int) -> bool:
    return 0 < (l - p) <= (1 << max(nu2(p + 1), nu2(l + 1)))


@lru_cache(maxsize=None)
def _dodis_support(m: int) -> tuple[tuple[int, int], ...]:
    if m < 0:
        raise ValueError(f"m must be nonnegative, got {m}")
    pairs = []
    for p in range(m + 1):
        for l in range(p + 1, m + 1):
            if _satisfies_eq1(p, l):
                pairs.append((p, l))
    return tuple(pairs)


def dodis_pairs(m: int) -> list[tuple[int, int]]:
    """Enumerate the support of 𝒟 on bids {0, ..., m}."""
    return list(_dodis_support(m))


def sample_dodis_pair(rng: np.random.Generator, m: int) -> tuple[int, int]:
    """Draw one pair uniformly from the support of 𝒟 on {0, ..., m}."""
    support: Sequence[tuple[int, int]] = _dodis_support(m)
    if not support:
        raise ValueError(f"Dodis support is empty for m={m}")
    idx = int(rng.integers(0, len(support)))
    return support[idx]
