"""Hand checks for Cantelli and layer bounds."""

import math

import pytest

from lea_ptq.bound import cantelli_bound, eps_total, layer_bound


def test_cantelli_bound_matches_hand():
    # delta=0.2, sqrt((1-0.2)/0.2) = 2. Bound = 2 + 3*2 = 8.
    assert cantelli_bound(2.0, 3.0, 0.2) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        cantelli_bound(0.0, -0.1, 0.2)
    with pytest.raises(ValueError):
        cantelli_bound(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        cantelli_bound(0.0, 1.0, 1.0)


def test_layer_bound_includes_covariance():
    # m = 1*2 + 3*1 = 5
    # v = 2**2 * 2**2 + 1**2 * 4**2 + 2*2*1*0.5 = 16 + 16 + 2 = 34
    # (1-0.2)/0.2 = 4, so bound = 5 + sqrt(136) = 5 + 2*sqrt(34)
    got = layer_bound(1.0, 2.0, 3.0, 4.0, 0.5, 2.0, 1.0, 0.2)
    assert got == pytest.approx(5.0 + 2.0 * math.sqrt(34.0))


def test_eps_total_sums_pairs():
    assert eps_total(0.1, [(0.2, 0.3), (0.4, 0.05)]) == pytest.approx(1.05)
