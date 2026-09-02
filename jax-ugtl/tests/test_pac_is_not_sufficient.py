"""Section 3.1: a surrogate can be right on almost every design and still send
greedy somewhere arbitrarily bad, while the same error budget spent on the
trajectory keeps the Theorem 4 guarantee.

This is the motivation for algorithm-dependent learnability, so it gets its own
test rather than a sentence in the README.
"""

import numpy as np
import pytest

from ugtl.theory import (
    HiddenJackpotCoverage,
    PerturbedOracle,
    alpha_greedy,
    greedy,
)


def run_greedy_on(value_fn, n, k):
    oracle = PerturbedOracle(value_fn, 0.0)
    return greedy(oracle, n, k)


@pytest.mark.parametrize("block", [10, 100, 1000, 10_000])
def test_greedy_on_a_mostly_accurate_surrogate_is_arbitrarily_bad(block):
    inst = HiddenJackpotCoverage(n=500, k=3, block=block)
    X = run_greedy_on(inst.surrogate, inst.n, inst.k)

    # Greedy took decoys, so its ratio is 1 / block and goes to zero with block
    # while the surrogate's error rate stays put.
    assert inst.f(X) / inst.optimum == pytest.approx(1.0 / block)
    assert inst.mismatch_rate(np.random.default_rng(0)) < 0.02


def test_greedy_on_the_true_function_is_optimal_here():
    """The instance is optimizable; only the learned function is the problem."""
    inst = HiddenJackpotCoverage(n=500, k=3, block=100)
    X = run_greedy_on(inst.f, inst.n, inst.k)
    assert inst.f(X) == inst.optimum


def test_the_surrogate_is_exact_off_the_optimal_region():
    inst = HiddenJackpotCoverage(n=500, k=3, block=100)
    rng = np.random.default_rng(1)
    wrong = 0
    for _ in range(2000):
        X = frozenset(rng.choice(inst.n, size=inst.k, replace=False).tolist())
        if X & inst.jackpots:
            wrong += 1
            assert inst.surrogate(X) < inst.f(X) / 10.0
        else:
            assert inst.surrogate(X) == inst.f(X)
    assert 0 < wrong < 60


@pytest.mark.parametrize("block", [10, 100, 1000, 10_000])
def test_the_same_error_budget_on_the_trajectory_keeps_the_guarantee(block):
    """Definition 4 with beta equal to the PAC error rate, on the same instance.

    The surrogate above is wrong on under 2 percent of designs and gives a ratio
    of 1 / block. A surrogate that is instead within (1 +/- 0.02) at every set
    greedy evaluates, chosen adversarially, still clears alpha of Theorem 4.
    """
    inst = HiddenJackpotCoverage(n=500, k=3, block=block)
    beta = 0.02
    oracle = PerturbedOracle(inst.f, beta, policy="adversarial")
    X = greedy(oracle, inst.n, inst.k)
    assert oracle.condition_holds()

    alpha = alpha_greedy(beta, 1.0, inst.k)
    assert alpha > 0.65
    assert inst.f(X) >= alpha * inst.optimum
    assert inst.f(X) / inst.optimum > 1.0 / block
