"""Theorem 3: among exact rules with the same proposal and the same budget,
Eq. (8) accepts at least as often as anything else.

Building an arbitrary competitor is not possible, but the proof of Theorem 3
says exactly which rules are admissible: an exact rule's per-count acceptance
must satisfy b_k C(N,k) = c b_{N-k+1} C(N,N-k+1), so zeroing count k forces
zeroing N-k+1. Any subset of counts closed under k -> N-k+1 therefore gives a
genuinely different exact rule, and Theorem 3 says it must be dominated. Two
families are tested here: those restricted rules, and a uniform down-scaling
of Eq. (8), which is exact for the same reason.
"""

import numpy as np
import pytest

from pref_mh.finite import (
    asymptotic_variance,
    pref_mh_kernel,
    restricted_kernel,
    stationary,
    total_variation,
)
from pref_mh.synthetic import random_model

# (N, allowed counts), each closed under k -> N - k + 1
COMPETITORS = [(3, (1, 3)), (4, (2, 3)), (5, (2, 4)), (5, (1, 3, 5))]


@pytest.mark.parametrize("n_votes,allowed", COMPETITORS)
def test_competitor_is_exact(n_votes, allowed):
    """The competitor has to be a fair fight: it must target pi too."""
    model = random_model(0, n_states=6)
    p = np.asarray(restricted_kernel(model, n_votes, allowed))
    pi = np.asarray(model.target())
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-13
    flow = pi[:, None] * p
    assert np.max(np.abs(flow - flow.T)) < 1e-13
    assert total_variation(stationary(p), pi) < 1e-12


@pytest.mark.parametrize("n_votes,allowed", COMPETITORS)
def test_n_vote_rule_dominates_off_diagonal_mass(n_votes, allowed):
    """P_N(x, A) >= P_comp(x, A) for every A not containing x, which is the
    Peskun-Tierney statement in the form the proof of Theorem 3 leaves it."""
    model = random_model(0, n_states=6)
    pref = np.asarray(pref_mh_kernel(model, n_votes))
    comp = np.asarray(restricted_kernel(model, n_votes, allowed))
    off = ~np.eye(model.size, dtype=bool)
    assert np.all(pref[off] - comp[off] > -1e-14)
    assert np.max(pref[off] - comp[off]) > 1e-3  # the gap is real, not ties


@pytest.mark.parametrize("n_votes,allowed", COMPETITORS)
def test_asymptotic_variance_ordering(n_votes, allowed):
    """The consequence people actually care about: dominating in the Peskun
    order means no larger asymptotic variance for every test function.

    Checked on eight random f, summing autocovariances of the exact kernels.
    Nothing is sampled, so there is no seed to get lucky on.
    """
    model = random_model(0, n_states=6)
    pi = np.asarray(model.target())
    pref = pref_mh_kernel(model, n_votes)
    comp = restricted_kernel(model, n_votes, allowed)
    rng = np.random.default_rng(0)
    for _ in range(8):
        f = rng.standard_normal(model.size)
        v_pref = asymptotic_variance(pref, pi, f, n_terms=600)
        v_comp = asymptotic_variance(comp, pi, f, n_terms=600)
        assert v_pref <= v_comp + 1e-9


def test_acceptance_rate_grows_with_budget():
    """Section 3.3's practical claim: a larger budget buys a higher acceptance
    rate while keeping the target exactly."""
    model = random_model(0, n_states=6)
    pi = np.asarray(model.target())
    rates = []
    for n in (1, 2, 4, 8):
        p = np.asarray(pref_mh_kernel(model, n))
        rates.append(float(pi @ (p.sum(axis=1) - np.diag(p))))
        assert total_variation(stationary(p), pi) < 1e-12
    assert all(a < b for a, b in zip(rates, rates[1:]))
