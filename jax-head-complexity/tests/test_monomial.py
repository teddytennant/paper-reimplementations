"""Theorem 5: monomial vote, and the Mobius expansion it rests on."""

import numpy as np

from head_complexity.monomial import (
    error_bound,
    evaluate,
    head_count,
    mobius_coefficients,
    monomial_vote_layer,
    truth_table_from_mobius,
    truth_table_of,
)


def test_mobius_of_3bit_parity():
    f = truth_table_of(lambda x: int(sum(x) % 2), 3)
    alpha = mobius_coefficients(f)
    assert alpha[()] == 0
    assert alpha[(0,)] == 1
    assert alpha[(1,)] == 1
    assert alpha[(2,)] == 1
    assert alpha[(0, 1)] == -2
    assert alpha[(0, 2)] == -2
    assert alpha[(1, 2)] == -2
    assert alpha[(0, 1, 2)] == 4


def test_mobius_roundtrip_random_boolean():
    rng = np.random.default_rng(0)
    n = 3
    for _ in range(5):
        f = rng.integers(0, 2, size=1 << n, dtype=np.int64)
        recovered = truth_table_from_mobius(mobius_coefficients(f), n)
        np.testing.assert_array_equal(recovered, f)


def test_monomial_vote_and_or_parity():
    n = 3
    targets = {
        "and": truth_table_of(lambda x: int(all(x)), n),
        "or": truth_table_of(lambda x: int(any(x)), n),
        "parity": truth_table_of(lambda x: int(sum(x) % 2), n),
    }
    for name, table in targets.items():
        layer = monomial_vote_layer(table)
        out = evaluate(layer, n)
        np.testing.assert_array_equal(out, table, err_msg=name)
        assert 1 <= head_count(table) <= 1 << n
        assert error_bound(table) < 0.5
