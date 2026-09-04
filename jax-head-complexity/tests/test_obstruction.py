"""Theorem 1: the alternating-sum obstruction."""

from head_complexity.obstruction import (
    parity_alternating_sum,
    random_argmax_layer,
    relative_alternating_sum,
)
from head_complexity.parity import parity_layer


def test_theorem_1_parity_layer_vanishes_when_n_exceeds_k():
    for k in (1, 2, 3):
        n = k + 1
        ratio = relative_alternating_sum(parity_layer(k, n=n), n)
        assert ratio == 0 or abs(ratio) < 1e-8, f"k={k} n={n} ratio={ratio}"


def test_theorem_1_random_layers_vanish_when_n_exceeds_k():
    cases = [(0, 1, 2), (1, 2, 3), (2, 2, 4), (3, 3, 4)]
    for seed, k, n in cases:
        layer = random_argmax_layer(seed, k, n)
        ratio = relative_alternating_sum(layer, n)
        assert abs(ratio) < 1e-8, f"seed={seed} k={k} n={n} ratio={ratio}"


def test_construction_alternating_sum_positive_at_n_equals_k():
    for k in (1, 2, 3):
        ratio = parity_alternating_sum(k, k)
        assert ratio > 0.5, f"k={k} ratio={ratio}"
