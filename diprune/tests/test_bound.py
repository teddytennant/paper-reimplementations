"""Equation checks for the token-wise pruning bound."""

import itertools

import jax
import numpy as np
import pytest

from diprune import inner_bound, relaxed_bound, token_weights, topk_mask

jax.config.update("jax_enable_x64", True)


def test_weights_and_inner_bound_match_the_formulas():
    v = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    g = np.array([[0.0, 1.0], [3.0, 0.0], [-1.0, 2.0]])
    mu = 0.5
    l_f = 1.0
    mask = np.array([1.0, 0.0, 1.0])
    g_sq = np.sum(g * g, axis=1)
    v_sq = np.sum(v * v, axis=1)
    expected_w = (1.0 / (2.0 * mu)) * g_sq + ((mu + l_f) / 2.0) * v_sq
    np.testing.assert_allclose(token_weights(v, g, mu, l_f), expected_w)
    inner = np.sum(g * v, axis=1)
    per_token = np.abs(inner) + (l_f / 2.0) * v_sq
    expected_inner = np.sum((1.0 - mask) * per_token)
    expected_relaxed = np.sum((1.0 - mask) * expected_w)
    assert float(inner_bound(v, g, mask, l_f)) == pytest.approx(expected_inner)
    assert float(relaxed_bound(v, g, mask, mu, l_f)) == pytest.approx(expected_relaxed)


def test_relaxed_bound_dominates_inner_bound():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(8, 5))
    g = rng.normal(size=(8, 5))
    mask = rng.integers(0, 2, size=8)
    mu = 0.7
    l_f = 1.3
    b_inner = float(inner_bound(v, g, mask, l_f))
    b_relaxed = float(relaxed_bound(v, g, mask, mu, l_f))
    assert b_relaxed >= b_inner - 1e-5


def test_topk_mask_minimizes_relaxed_bound():
    rng = np.random.default_rng(1)
    n, k = 6, 2
    v = rng.normal(size=(n, 4))
    g = rng.normal(size=(n, 4))
    mu = 0.8
    l_f = 1.1
    weights = np.asarray(token_weights(v, g, mu, l_f))
    mask = np.asarray(topk_mask(weights, k))
    got = float(relaxed_bound(v, g, mask, mu, l_f))
    best = None
    for combo in itertools.combinations(range(n), k):
        trial = np.zeros(n)
        trial[list(combo)] = 1.0
        value = float(relaxed_bound(v, g, trial, mu, l_f))
        if best is None or value < best:
            best = value
    assert got == pytest.approx(best)
    assert int(mask.sum()) == k


def test_larger_weight_is_preferred_when_k_is_1():
    weights = np.array([0.2, 1.5, 0.7, 0.1])
    mask = np.asarray(topk_mask(weights, 1))
    assert mask.tolist() == [0, 1, 0, 0]


def test_weight_tie_keeps_the_smaller_index():
    weights = np.array([0.4, 0.4, 0.1])
    mask = np.asarray(topk_mask(weights, 1))
    assert mask.tolist() == [1, 0, 0]
