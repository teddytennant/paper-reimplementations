"""Valid prices (Definition 10) and Myerson reconstruction."""

from __future__ import annotations

import numpy as np

from ic_tester import is_valid_price_1d, jumps, myerson_price_1d


def test_myerson_price_valid_on_step():
    # Jump at t=3: x = 0,0,0,1,1,1,2,2  (m=7).
    x = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    assert jumps(x) == [0, 3, 6]
    p = myerson_price_1d(x)
    assert is_valid_price_1d(x, p)
    assert p[0] == 0
    # Constant on plateaus of x.
    assert np.all(p[0:3] == p[0])
    assert np.all(p[3:6] == p[3])
    assert np.all(p[6:] == p[6])
    # Left-endpoint multipliers: h_1 = 2, h_2 = 5.
    assert p[3] == 2.0
    assert p[6] == 2.0 + 5.0


def test_zero_price_invalid_when_jump_at_least_2():
    x = np.array([0, 0, 1, 1, 1])  # jump at t=2, dx=1 > 0
    p = np.zeros_like(x, dtype=float)
    assert not is_valid_price_1d(x, p)
