"""Theorem 4: greedy on a surrogate that is only accurate along its own
trajectory still clears the alpha the theorem states."""

import math

import numpy as np
import pytest

from ugtl.theory import (
    FacilityLocation,
    MaxCoverage,
    PerturbedOracle,
    PowerModular,
    alpha_greedy,
    brute_force_max,
    greedy,
    greedy_value_oracle_ratio,
    submodularity_ratio,
)

BETAS = [0.0, 0.02, 0.05, 0.1, 0.2]


def random_coverage(rng, n=10, universe=14):
    """A random max coverage instance, monotone and submodular."""
    cover = [rng.choice(universe, size=int(rng.integers(1, 5)), replace=False).tolist() for _ in range(n)]
    return MaxCoverage(cover)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("policy", ["adversarial", "random"])
def test_greedy_ratio_holds_on_max_coverage(beta, policy):
    k, n = 3, 10
    for seed in range(10):
        rng = np.random.default_rng(seed)
        f = random_coverage(rng, n=n)
        oracle = PerturbedOracle(f, beta, policy=policy, seed=seed)
        X = greedy(oracle, n, k)

        # The surrogate really does satisfy Eq. (3) on everything greedy touched.
        assert oracle.condition_holds()

        gamma = submodularity_ratio(f, X, k, n)
        _, opt = brute_force_max(f, n, k)
        assert f(X) >= alpha_greedy(beta, gamma, k) * opt - 1e-9


@pytest.mark.parametrize("beta", BETAS)
def test_greedy_ratio_holds_on_facility_location(beta):
    """A second submodular family, in case max coverage's integer values make the
    adversary's job too easy."""
    k, n = 3, 9
    for seed in range(8):
        rng = np.random.default_rng(50 + seed)
        f = FacilityLocation(rng.uniform(0.0, 1.0, (n, 12)))
        oracle = PerturbedOracle(f, beta, policy="adversarial", seed=seed)
        X = greedy(oracle, n, k)
        assert oracle.condition_holds()
        gamma = submodularity_ratio(f, X, k, n)
        assert gamma == pytest.approx(1.0)
        _, opt = brute_force_max(f, n, k)
        assert f(X) >= alpha_greedy(beta, gamma, k) * opt - 1e-9


@pytest.mark.parametrize("beta", [0.0, 0.02, 0.05])
def test_greedy_ratio_holds_when_only_approximately_submodular(beta):
    """PowerModular has gamma < 1, so the gamma_{X_k,k} in the bound matters."""
    n, k = 8, 3
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        f = PowerModular(rng.uniform(0.2, 1.0, n), p=1.4)
        oracle = PerturbedOracle(f, beta, policy="adversarial", seed=seed)
        X = greedy(oracle, n, k)
        gamma = submodularity_ratio(f, X, k, n)
        assert gamma < 1.0
        _, opt = brute_force_max(f, n, k)
        assert f(X) >= alpha_greedy(beta, gamma, k) * opt - 1e-9


def test_submodularity_ratio_is_one_for_submodular_functions():
    rng = np.random.default_rng(0)
    f = random_coverage(rng, n=8)
    for _ in range(5):
        X = rng.choice(8, size=3, replace=False).tolist()
        assert submodularity_ratio(f, X, 3, 8) == pytest.approx(1.0)


def test_alpha_reduces_to_the_value_oracle_ratio_at_beta_zero():
    for k in [1, 2, 3, 5, 10, 25]:
        for gamma in [0.3, 0.6, 1.0]:
            assert alpha_greedy(0.0, gamma, k) == pytest.approx(greedy_value_oracle_ratio(gamma, k), rel=1e-12)


def test_alpha_tends_to_one_minus_one_over_e():
    """gamma = 1 and beta = 1/n^2 gives 1 - 1/e - o(1), the recoverability result."""
    target = 1.0 - 1.0 / math.e
    gaps = [target - alpha_greedy(1.0 / n**2, 1.0, n) for n in (50, 100, 400, 1000, 4000)]
    assert all(g > 0 for g in gaps)
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-4


def test_alpha_decreases_as_the_surrogate_gets_worse():
    values = [alpha_greedy(b, 1.0, 5) for b in (0.0, 0.05, 0.1, 0.3, 0.6)]
    assert values == sorted(values, reverse=True)
