"""Theorem 1 and the plug-in baseline it rules out.

Theorem 1: with a fixed budget of N binary votes, no procedure has marginal
acceptance equal to alpha_BT-MH(p) = min{1, c p/(1-p)} for every p in (0,1).
The proof is a polynomial argument, so it can be checked exactly instead of
sampled.

The contrast that matters: the plug-in rule of Appx. C.1 accepts on an
estimate of the odds and is a biased sampler, while Pref-MH spends the same
budget on the same votes and is exact.
"""

import math

import numpy as np
import pytest

from pref_mh.finite import plugin_kernel, pref_mh_kernel, stationary, total_variation
from pref_mh.synthetic import random_model

BUDGETS = [1, 2, 3, 4, 6, 8]


def _forward_map(n_votes):
    """Columns are the coefficients of (1-p) C(N,k) p^k (1-p)^{N-k}.

    Any fixed-budget rule accepts with probability a(z) after seeing the vote
    vector z. Its marginal is A_N(p) = sum_z a(z) p^{|z|}(1-p)^{N-|z|}, which
    depends on z only through |z|, so writing b_k for the average over vectors
    with k ones gives A_N(p) = sum_k b_k C(N,k) p^k (1-p)^{N-k}. Multiplying
    by (1-p) gives this matrix, of shape (N+2, N+1) in the monomial basis.
    """
    cols = []
    for k in range(n_votes + 1):
        poly = np.zeros(1)
        poly[0] = math.comb(n_votes, k)
        for _ in range(k):
            poly = np.convolve(poly, [0.0, 1.0])
        for _ in range(n_votes - k + 1):
            poly = np.convolve(poly, [1.0, -1.0])
        cols.append(np.pad(poly, (0, n_votes + 2 - len(poly))))
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("n_votes", BUDGETS)
@pytest.mark.parametrize("c", [0.25, 1.0, 3.7])
def test_no_fixed_budget_rule_matches_the_oracle(n_votes, c):
    """The system (1-p) A_N(p) = c p has no solution in b, for any real b.

    On the interval where c p/(1-p) < 1 the oracle rule equals c p/(1-p), so a
    matching procedure would force the polynomial identity above. Every column
    of the map carries a factor (1-p) and so sums to zero at p = 1, while the
    right-hand side is c there. That is the paper's B(1) = -c step, and it
    kills the system regardless of whether b is constrained to [0,1].
    """
    a = _forward_map(n_votes)
    target = np.zeros(n_votes + 2)
    target[1] = c

    assert np.max(np.abs(a.sum(axis=0))) < 1e-10  # every column vanishes at p=1
    assert abs(target.sum() - c) < 1e-12

    b, *_ = np.linalg.lstsq(a, target, rcond=None)
    assert np.linalg.norm(a @ b - target) > 0.1 * c


@pytest.mark.parametrize("n_votes", [1, 2, 3, 4])
def test_plugin_is_biased_where_pref_mh_is_not(n_votes):
    """Appx. C.1 baseline against Eq. (8), same votes, same budget.

    Pref-MH's stationary distribution is pi to 1e-12. The plug-in's is off by
    a TV distance you can see, and it misses detailed balance outright.
    """
    model = random_model(0, n_states=6)
    pi = np.asarray(model.target())

    pref = stationary(pref_mh_kernel(model, n_votes))
    plug = np.asarray(plugin_kernel(model, n_votes))

    assert total_variation(pref, pi) < 1e-12
    assert total_variation(stationary(plug), pi) > 0.02

    flow = pi[:, None] * plug
    assert np.max(np.abs(flow - flow.T)) > 1e-3


def test_plugin_bias_shrinks_with_budget_but_does_not_vanish():
    """The plug-in odds estimate is consistent, so more votes help, but the
    bias is still there at N = 8 while Pref-MH is exact at N = 1."""
    model = random_model(0, n_states=6)
    pi = np.asarray(model.target())
    tvs = [
        total_variation(stationary(plugin_kernel(model, n)), pi)
        for n in (1, 2, 4, 8)
    ]
    assert tvs[0] > tvs[-1]
    assert tvs[-1] > 1e-3
    assert total_variation(stationary(pref_mh_kernel(model, 1)), pi) < 1e-12
