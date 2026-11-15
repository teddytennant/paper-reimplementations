"""Theorem 2 and Corollary 1: the Pref-MH chain has pi as its stationary
distribution, exactly, for every fixed budget N.

The finite state spaces here are tiny on purpose. Every acceptance probability
is the exhaustive binomial sum of Eq. (13), not a Monte Carlo estimate, so
these tests carry no sampling error and can be asserted at 1e-12. None of them
depend on the chain mixing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pref_mh.acceptance import marginal_acceptance, marginal_acceptance_multi
from pref_mh.finite import (
    oracle_mh_kernel,
    pref_mh_kernel,
    stationary,
    total_variation,
)
from pref_mh.synthetic import random_model

BUDGETS = [1, 2, 3, 5]
SEEDS = [0, 1, 2]


def _kernel_is_valid(p):
    p = np.asarray(p)
    assert np.all(p >= -1e-15)
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-13


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_votes", BUDGETS)
def test_target_is_stationary(seed, n_votes):
    """pi @ P_N = pi, by exhaustive enumeration of Eq. (14).

    The matmul runs in float64; see tests/test_precision.py for why that is
    checked rather than assumed.
    """
    model = random_model(seed, n_states=6)
    p = pref_mh_kernel(model, n_votes)
    pi = model.target()
    _kernel_is_valid(p)
    assert p.dtype == jnp.dtype("float64")
    assert float(jnp.max(jnp.abs(pi @ p - pi))) < 1e-12


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_votes", BUDGETS)
def test_detailed_balance(seed, n_votes):
    """Eq. (9): pi_i P_ij = pi_j P_ji on the same finite space."""
    model = random_model(seed, n_states=6)
    p = pref_mh_kernel(model, n_votes)
    flow = model.target()[:, None] * p
    assert float(jnp.max(jnp.abs(flow - flow.T))) < 1e-13


@pytest.mark.parametrize("n_votes", BUDGETS)
def test_marginal_acceptance_ratio_identity(n_votes):
    """The core step of the proof of Theorem 2, checked directly.

    bar-alpha_N(x,y) / bar-alpha_N(y,x) = pi(y)q(x|y) / (pi(x)q(y|x)), with
    both sides computed independently: the left from the binomial sum of
    Eq. (13), the right from the densities. This is detailed balance before it
    is assembled into a matrix.
    """
    model = random_model(4, n_states=7, symmetric=False)
    lr = model.log_r0()
    w = model.win_probs()[..., 0]
    fwd = marginal_acceptance(lr, w, n_votes)
    rev = marginal_acceptance(lr.T, w.T, n_votes)
    lt = np.asarray(model.log_target())
    lq = np.asarray(model.log_q)
    want = (lt[None, :] + lq) - (lt[:, None] + lq.T)
    got = np.log(np.asarray(fwd)) - np.log(np.asarray(rev))
    assert np.allclose(got, want, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n_judges", [2, 3])
@pytest.mark.parametrize("n_votes", [1, 2, 3])
def test_multi_judge_exactness(n_judges, n_votes):
    """Theorem 4 / Eq. (12): the composite target pi_m of Eq. (10) is
    stationary and reversible when each judge contributes its own N-vote
    factor."""
    model = random_model(7, n_states=5, n_judges=n_judges)
    p = pref_mh_kernel(model, n_votes)
    pi = model.target()
    _kernel_is_valid(p)
    assert float(jnp.max(jnp.abs(pi @ p - pi))) < 1e-12
    flow = pi[:, None] * p
    assert float(jnp.max(jnp.abs(flow - flow.T))) < 1e-13


def test_multi_judge_reduces_to_single_judge():
    """Eq. (12) at m = 1 is Eq. (8)."""
    model = random_model(9, n_states=5, n_judges=1)
    lr = model.log_r0()
    w = model.win_probs()
    for n in BUDGETS:
        a = marginal_acceptance(lr, w[..., 0], n)
        b = marginal_acceptance_multi(lr, w, n)
        assert float(jnp.max(jnp.abs(a - b))) < 1e-14


@pytest.mark.parametrize("n_votes", BUDGETS)
def test_stationary_vector_matches_oracle_mh(n_votes):
    """Pref-MH and standard MH on the true density land on the same
    distribution, though not the same kernel.

    Solved as a left null space, so this does not depend on how fast either
    chain mixes.
    """
    model = random_model(1, n_states=6)
    pi = np.asarray(model.target())
    pref = stationary(pref_mh_kernel(model, n_votes))
    oracle = stationary(oracle_mh_kernel(model))
    assert total_variation(pref, pi) < 1e-12
    assert total_variation(oracle, pi) < 1e-12
    assert total_variation(pref, oracle) < 1e-12
