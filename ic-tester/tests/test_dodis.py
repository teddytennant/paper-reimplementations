"""Dodis pair distribution (Eq. (1) / Lemma 3.8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ic_tester import dodis_pairs, nu2, sample_dodis_pair


def _eq1(p: int, l: int) -> bool:
    def nu(r: int) -> int:
        t = 0
        while r % 2 == 0:
            r //= 2
            t += 1
        return t

    return 0 < (l - p) <= 2 ** max(nu(p + 1), nu(l + 1))


def test_nu2_values():
    assert nu2(1) == 0
    assert nu2(2) == 1
    assert nu2(3) == 0
    assert nu2(4) == 2
    assert nu2(8) == 3
    assert nu2(12) == 2


@pytest.mark.parametrize("m", [1, 2, 3, 7, 8])
def test_dodis_support_matches_eq1(m: int):
    support = dodis_pairs(m)
    support_set = set(support)
    brute = []
    for p in range(m + 1):
        for l in range(m + 1):
            if _eq1(p, l):
                brute.append((p, l))
    assert support == brute
    assert support_set == set(brute)
    assert all(l > p for p, l in support)
    assert all(_eq1(p, l) for p, l in support)
    for p in range(m + 1):
        for l in range(m + 1):
            if _eq1(p, l):
                assert (p, l) in support_set
            else:
                assert (p, l) not in support_set


def test_sample_dodis_pair_uniform_on_m3():
    m = 3
    support = dodis_pairs(m)
    k = len(support)
    rng = np.random.default_rng(0)
    n_draws = 20000
    counts = {pair: 0 for pair in support}
    for _ in range(n_draws):
        pair = sample_dodis_pair(rng, m)
        counts[pair] += 1
    expected = n_draws / k
    # Chi-square goodness of fit against uniform.
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    # df = k-1 = 4; critical value at 0.001 is about 18.5.
    assert chi2 < 18.5
    max_rel = max(abs(c - expected) / expected for c in counts.values())
    assert max_rel < 0.15
    # Every support pair appears; nothing outside the support.
    assert all(c > 0 for c in counts.values())
    assert math.isclose(sum(counts.values()), n_draws)
