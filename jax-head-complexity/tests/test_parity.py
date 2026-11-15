"""Theorem 2: k heads compute k-bit parity."""

import numpy as np

from head_complexity.layer import all_inputs
from head_complexity.parity import (
    DEFAULT_MASK,
    construction_margin,
    gap_at_weight,
    parity,
    parity_layer,
)


def test_parity_exact_on_all_inputs():
    for k in (1, 2, 3, 4):
        layer = parity_layer(k)
        xs = all_inputs(k)
        out = np.asarray(layer.decide(xs))
        target = parity(xs)
        assert (out == target).all(), f"k={k} failed on some of 2^{k} inputs"


def test_delta_has_no_ties_and_sign_matches_parity():
    for k in (1, 2, 3, 4):
        layer = parity_layer(k)
        xs = np.asarray(all_inputs(k))
        delta = np.asarray(layer.delta(xs))
        assert np.min(np.abs(delta)) > 0.0, f"k={k} has a tie"
        c = xs.sum(axis=-1)
        expected = np.where(c % 2 == 0, 1.0, -1.0)
        np.testing.assert_array_equal(np.sign(delta), expected)


def test_gap_at_weight_sign_lemma_4():
    for k in (1, 2, 3, 4):
        for c in range(k + 1):
            g = gap_at_weight(k, c)
            assert np.sign(g) == (-1) ** c, f"k={k} c={c} gap={g}"


def test_denominators_on_all_zeros():
    for k in (1, 2, 3, 4):
        layer = parity_layer(k)
        zeros = np.zeros((1, k), dtype=np.int32)
        d_x, _ = layer.stats(zeros)
        expected = np.arange(1, k + 1, dtype=np.float64) + np.exp(-DEFAULT_MASK)
        np.testing.assert_allclose(np.asarray(d_x)[0], expected, atol=1e-6)


def test_construction_margin_positive():
    for k in (1, 2, 3, 4):
        assert construction_margin(k) > 0.0
