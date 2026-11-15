"""Exact checks on the weight estimator.

Every claim the paper makes about w = 1 - p_hat is a statement about a sum over
the K+1 outcomes of K Bernoulli(p) rollouts, so these tests enumerate that sum
in exact rational arithmetic. Nothing here samples.
"""

import math
from fractions import Fraction

import jax.numpy as jnp
import jax.random as jr
import pytest

from osw_ft.weighting import (
    expected_squared_weight,
    expected_weight,
    hard_threshold_weight,
    osw_weight,
    p_hat,
    random_weight,
    signal_loss_probability,
    sigma_squared,
    sigma_squared_sft,
    weight_pmf,
)

P_GRID = [Fraction(i, 8) for i in range(9)]
K_GRID = [1, 2, 3, 4, 5]


def binomial_prob(p, K, m):
    """Exact P(m of K rollouts correct)."""
    return Fraction(math.comb(K, m)) * p**m * (1 - p) ** (K - m)


def enumerated_weight_moment(p, K, power):
    """Exact E[(1 - p_hat)^power] by summing over all K+1 rollout outcomes."""
    return sum(
        binomial_prob(p, K, m) * (1 - Fraction(m, K)) ** power for m in range(K + 1)
    )


@pytest.mark.parametrize("K", K_GRID)
@pytest.mark.parametrize("p", P_GRID)
def test_theorem1_unbiasedness_exact(p, K):
    """Theorem 1: E_p_hat[(1 - p_hat) g_0] = (1 - p) g_0 at every finite K.

    g_0 is deterministic given q, so it factors out and the claim is
    E[1 - p_hat] = 1 - p. Checked as an equality of Fractions, not a tolerance.
    """
    assert enumerated_weight_moment(p, K, 1) == 1 - p
    assert float(expected_weight(float(p), K)) == pytest.approx(float(1 - p), abs=0.0)


@pytest.mark.parametrize("K", K_GRID)
def test_theorem1_holds_for_a_real_gradient_vector(K):
    """Same statement carried through an actual g_0 vector rather than a scalar."""
    p = 0.375
    g0 = jnp.asarray([1.5, -2.0, 0.25, 7.0])
    values, probs = weight_pmf(p, K)
    assert float(jnp.sum(probs)) == pytest.approx(1.0, abs=1e-15)
    estimator_mean = jnp.sum(probs[:, None] * values[:, None] * g0[None, :], axis=0)
    assert jnp.allclose(estimator_mean, (1.0 - p) * g0, rtol=0, atol=1e-14)


@pytest.mark.parametrize("K", K_GRID)
@pytest.mark.parametrize("p", P_GRID)
def test_second_moment_identity_exact(p, K):
    """Remark 2, eq (14): E[(1 - p_hat)^2] = (1-p)^2 + p(1-p)/K, exactly."""
    closed_form = (1 - p) ** 2 + p * (1 - p) / K
    assert enumerated_weight_moment(p, K, 2) == closed_form
    assert float(expected_squared_weight(float(p), K)) == pytest.approx(
        float(closed_form), rel=1e-14
    )


@pytest.mark.parametrize("K", K_GRID)
@pytest.mark.parametrize("p", P_GRID)
def test_estimation_mse_exact(p, K):
    """Remark 3, eq (15): E[(p - p_hat)^2] = Var[p_hat] = p(1-p)/K."""
    enumerated = sum(
        binomial_prob(p, K, m) * (p - Fraction(m, K)) ** 2 for m in range(K + 1)
    )
    assert enumerated == p * (1 - p) / K


@pytest.mark.parametrize("K", K_GRID)
@pytest.mark.parametrize("p", P_GRID)
def test_remark1_signal_loss_probability_exact(p, K):
    """Remark 1, eq (13): P(w = 0) = p^K.

    w = 0 iff p_hat = 1 iff all K rollouts are correct.
    """
    enumerated = sum(
        binomial_prob(p, K, m)
        for m in range(K + 1)
        if 1 - Fraction(m, K) == 0
    )
    assert enumerated == p**K
    assert float(signal_loss_probability(float(p), K)) == pytest.approx(float(p**K), rel=1e-14)


def test_remark1_transition_at_k_two():
    """The numbers Remark 1 quotes for p = 0.5: 50% signal loss at K=1, 25% at K=2.

    And the intermediate weight w = 0.5 that makes soft discrimination possible
    only exists once K >= 2.
    """
    half = Fraction(1, 2)
    assert signal_loss_probability(0.5, 1) == 0.5
    assert signal_loss_probability(0.5, 2) == 0.25
    assert set(weight_pmf(0.5, 1)[0].tolist()) == {0.0, 1.0}
    assert 0.5 in weight_pmf(0.5, 2)[0].tolist()
    assert enumerated_weight_moment(half, 2, 1) == half


@pytest.mark.parametrize("K", K_GRID)
@pytest.mark.parametrize("p", [p for p in P_GRID if p > 0])
def test_corollary1_integrand_strictly_positive(p, K):
    """Corollary 1: the per-query gap p(2-p) - p(1-p)/K is strictly positive.

    Exact rational check, so the strictness is not a floating point artifact.
    """
    gap = p * (2 - p) - p * (1 - p) / K
    assert gap > 0
    assert gap >= p  # the bound used in eq (22)


@pytest.mark.parametrize("K", K_GRID)
def test_sigma_squared_below_sft(K):
    """eq (11): Sigma^2 < Sigma^2_SFT whenever p > 0 somewhere with positive measure."""
    p = jnp.asarray([0.0, 0.0, 0.1, 0.5, 0.9, 1.0])
    g0_sq = jnp.asarray([3.0, 1.0, 2.5, 0.75, 4.0, 1.25])
    assert float(sigma_squared(p, g0_sq, K)) < float(sigma_squared_sft(g0_sq))


@pytest.mark.parametrize("K", K_GRID)
def test_sigma_squared_equals_sft_when_never_correct(K):
    """With p = 0 everywhere the weight is identically 1 and OSW-FT is SFT."""
    p = jnp.zeros(5)
    g0_sq = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert float(sigma_squared(p, g0_sq, K)) == pytest.approx(
        float(sigma_squared_sft(g0_sq)), rel=1e-15
    )


def test_sigma_squared_matches_enumerated_second_moment():
    """Cross-check eq (11) against the enumerated E[w^2] for a small dataset."""
    K = 3
    ps = [Fraction(1, 4), Fraction(1, 2), Fraction(7, 8)]
    g0_sq = [2.0, 0.5, 3.25]
    enumerated = sum(
        float(enumerated_weight_moment(p, K, 2)) * n for p, n in zip(ps, g0_sq)
    ) / len(ps)
    got = float(sigma_squared(jnp.asarray([float(p) for p in ps]), jnp.asarray(g0_sq), K))
    assert got == pytest.approx(enumerated, rel=1e-13)


def test_p_hat_and_weight_from_rewards():
    """Algorithm 1 lines 6-7 on a batch of rewards."""
    rewards = jnp.asarray([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    assert jnp.allclose(p_hat(rewards), jnp.asarray([1.0, 0.0, 0.5]))
    assert jnp.allclose(osw_weight(rewards), jnp.asarray([0.0, 1.0, 0.5]))
    # section 4.4 control: the same rewards collapsed to a binary weight
    assert jnp.allclose(hard_threshold_weight(rewards), jnp.asarray([0.0, 1.0, 1.0]))


@pytest.mark.parametrize("mean", [0.2, 0.5, 0.85])
def test_random_weight_matches_target_mean(mean):
    """Section 4.4 control weights are calibrated to OSW-FT's mean weight."""
    w = random_weight(jr.PRNGKey(0), mean, (200_000,))
    assert float(jnp.min(w)) >= 0.0
    assert float(jnp.max(w)) <= 1.0
    assert float(jnp.mean(w)) == pytest.approx(mean, abs=5e-3)
