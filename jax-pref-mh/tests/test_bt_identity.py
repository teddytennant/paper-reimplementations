"""The paper's key observation, Eq. (6).

The BT preference odds equal the judge's likelihood ratio, so multiplying the
tractable factor r_0 by the odds reproduces the full MH ratio of Eq. (2). It
is a one-line identity and it should hold to machine precision.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pref_mh.acceptance import bt_log_odds, bt_win_prob, log_alpha_bt_mh
from pref_mh.finite import oracle_mh_kernel, pref_mh_kernel
from pref_mh.synthetic import random_model


def test_odds_equal_score_difference():
    """p/(1-p) = e^{s(y)-s(x)}, Eq. (6)."""
    rng = np.random.default_rng(0)
    sx = rng.normal(scale=3.0, size=4000)
    sy = rng.normal(scale=3.0, size=4000)
    p = bt_win_prob(jnp.asarray(sx), jnp.asarray(sy))
    got = np.asarray(bt_log_odds(p))
    want = sy - sx
    # Relative, not absolute: the round trip through sigma loses a few ulps of
    # the difference once p is within 1e-7 of 1, which is float64 arithmetic
    # rather than the identity failing.
    assert np.allclose(got, want, rtol=1e-11, atol=1e-13)


def test_odds_recover_the_mh_ratio():
    """r_0 * odds equals pi(y)q(x|y) / (pi(x)q(y|x)), Eq. (3) into Eq. (7).

    Done on a full finite instance so the proposal is asymmetric and the
    Hastings factor does not quietly cancel.
    """
    model = random_model(11, n_states=7, symmetric=False)
    lq = np.asarray(model.log_q)
    lt = np.asarray(model.log_target())
    want = (lt[None, :] + lq) - (lt[:, None] + lq.T)

    log_r0 = np.asarray(model.log_r0())
    odds = np.asarray(bt_log_odds(model.win_probs()[..., 0]))
    assert np.max(np.abs(log_r0 + odds - want)) < 1e-11


def test_bt_mh_rule_matches_standard_mh_kernel():
    """The oracle rule of Eq. (7), built only from preference probabilities,
    gives the same kernel as textbook MH on the true density, Eq. (2)."""
    model = random_model(5, n_states=6)
    from pref_mh.finite import kernel_from_acceptance

    alpha = jnp.exp(
        log_alpha_bt_mh(model.log_r0(), model.win_probs()[..., 0])
    )
    got = np.asarray(kernel_from_acceptance(model, alpha))
    want = np.asarray(oracle_mh_kernel(model))
    assert np.max(np.abs(got - want)) < 1e-12


@pytest.mark.parametrize("n_votes", [1, 2, 4])
def test_n_vote_rule_does_not_implement_the_oracle_rule(n_votes):
    """Theorem 1, seen from the other side.

    Pref-MH is exact but its marginal acceptance is not the oracle BT-MH
    acceptance, which is the whole content of Theorem 1: no fixed budget can
    hit that target. The gap has to be visible, not a rounding artefact.
    """
    model = random_model(2, n_states=6)
    pref = np.asarray(pref_mh_kernel(model, n_votes))
    oracle = np.asarray(oracle_mh_kernel(model))
    off = ~np.eye(model.size, dtype=bool)
    assert np.max(np.abs(pref - oracle)[off]) > 1e-3
