"""Theorem 5: local search on a trajectory-accurate surrogate clears alpha for
unconstrained non-monotone submodular maximization."""

import numpy as np
import pytest

from ugtl.theory import (
    MaxCut,
    PerturbedOracle,
    alpha_local_search,
    brute_force_max,
    local_search,
)


def random_max_cut(rng, n=8):
    W = rng.uniform(0.0, 1.0, (n, n))
    W = np.triu(W, 1)
    return MaxCut(W + W.T)


@pytest.mark.parametrize("beta", [0.0, 0.005, 0.01, 0.02])
@pytest.mark.parametrize("eps", [0.5, 2.0])
@pytest.mark.parametrize("policy", ["adversarial", "random"])
def test_local_search_ratio_holds_on_max_cut(beta, eps, policy):
    n = 8
    for seed in range(8):
        rng = np.random.default_rng(seed)
        f = random_max_cut(rng, n)
        oracle = PerturbedOracle(f, beta, policy=policy, seed=seed)
        X = local_search(oracle, n, eps)

        assert oracle.condition_holds()

        _, opt = brute_force_max(f, n)
        assert f(X) >= alpha_local_search(beta, eps, n) * opt - 1e-9


def test_alpha_reduces_to_one_over_three_plus_two_eps_over_n():
    for n in [4, 8, 20, 100]:
        for eps in [0.0, 0.1, 1.0, 5.0]:
            assert alpha_local_search(0.0, eps, n) == pytest.approx(1.0 / (3.0 + 2.0 * eps / n), rel=1e-12)


def test_alpha_is_a_little_better_than_one_third_minus_eps_over_n():
    """The paper notes alpha at beta = 0 beats the value-oracle ratio 1/3 - eps/n."""
    for n in [8, 20, 50]:
        for eps in [0.5, 2.0]:
            assert alpha_local_search(0.0, eps, n) > 1.0 / 3.0 - eps / n


def test_alpha_decreases_as_the_surrogate_gets_worse():
    values = [alpha_local_search(b, 1.0, 10) for b in (0.0, 0.01, 0.05, 0.1)]
    assert values == sorted(values, reverse=True)


def test_local_search_stops_at_a_surrogate_local_optimum():
    """Nothing in the returned set's neighbourhood improves f~ by 1 + eps/n^2."""
    n, eps, beta = 8, 1.0, 0.01
    rng = np.random.default_rng(3)
    f = random_max_cut(rng, n)
    oracle = PerturbedOracle(f, beta, policy="random", seed=3)
    X = local_search(oracle, n, eps)
    complement = frozenset(range(n)) - X
    local = X if oracle.cache[X] >= oracle.cache.get(complement, -np.inf) else complement
    fx = oracle.cache[local]
    for v in range(n):
        move = local | {v} if v not in local else local - {v}
        if move in oracle.cache:
            assert oracle.cache[move] <= (1.0 + eps / n**2) * fx + 1e-9
