"""Section 2 model, decision polynomial, and Theorem 8 read-out equivalence."""

import numpy as np

from head_complexity.layer import all_inputs, decision_polynomial, to_affine, to_argmax
from head_complexity.monomial import monomial_vote_layer, truth_table_of
from head_complexity.parity import parity_layer


def test_all_inputs_count():
    for n in (1, 2, 3, 4):
        xs = all_inputs(n)
        assert xs.shape == (1 << n, n)


def test_theorem_8_argmax_to_affine_on_parity():
    layer = parity_layer(3)
    xs = all_inputs(3)
    np.testing.assert_array_equal(
        np.asarray(to_affine(layer).decide(xs)),
        np.asarray(layer.decide(xs)),
    )


def test_theorem_8_affine_to_argmax_on_and():
    table = truth_table_of(lambda x: int(all(x)), 3)
    affine = monomial_vote_layer(table)
    xs = all_inputs(3)
    np.testing.assert_array_equal(
        np.asarray(to_argmax(affine).decide(xs)),
        np.asarray(affine.decide(xs)),
    )


def test_decision_polynomial_sign_matches_delta_and_G():
    argmax = parity_layer(3)
    xs = all_inputs(3)
    d_x, s_x = argmax.stats(xs)
    F = np.asarray(decision_polynomial(d_x, s_x))
    delta = np.asarray(argmax.delta(xs))
    np.testing.assert_array_equal(np.sign(F), np.sign(delta))

    table = truth_table_of(lambda x: int(all(x)), 3)
    affine = monomial_vote_layer(table)
    d_x, s_x = affine.stats(xs)
    F = np.asarray(decision_polynomial(d_x, s_x, theta=float(affine.theta)))
    G = np.asarray(affine.G(xs))
    np.testing.assert_array_equal(np.sign(F), np.sign(G))
