"""Proposition 4: the three probability designs are unbiased for L(Phi)."""

import jax
import jax.numpy as jnp
import pytest

from mc_util import (
    cached_stratified,
    cached_uniform,
    strat_pop,
    stratified_draws,
    summary,
    unif_pop,
)
from stratified_allocation import (
    balanced_allocation,
    class_counts,
    proportional_allocation,
    random_population,
)

REPS = 200_000


@pytest.mark.parametrize(
    "n, n1, budget",
    [(400, 40, 40), (500, 10, 60), (1000, 200, 100)],
)
def test_uniform_design_is_unbiased(n, n1, budget):
    pop = unif_pop(n, n1)
    draws = cached_uniform(f"u{n}-{n1}", 11, pop.losses, budget, REPS)
    mean, _, se = summary(draws)
    assert abs(float(mean - pop.total)) < 5.0 * float(se)


@pytest.mark.parametrize(
    "n, n1, budget",
    [(400, 40, 40), (500, 10, 60), (1000, 200, 100)],
)
@pytest.mark.parametrize("rule", ["proportional", "balanced"])
def test_class_conditional_designs_are_unbiased(n, n1, budget, rule):
    pop = strat_pop(n, n1)
    counts = class_counts(pop.labels)
    alloc = (
        proportional_allocation(counts, budget)
        if rule == "proportional"
        else balanced_allocation(counts, budget)
    )
    draws = cached_stratified(f"s{n}-{n1}", 12, pop.losses, pop.labels, *alloc, REPS)
    mean, _, se = summary(draws)
    assert abs(float(mean - pop.total)) < 5.0 * float(se)


def test_the_tolerance_is_actually_tight():
    """A guard on the test above: 5 standard errors has to be a small target.

    Without this, a design that returned the wrong estimand by a hair would pass simply
    because the Monte Carlo error was large.
    """
    pop = strat_pop(400, 40)
    draws = cached_stratified("s400-40", 12, pop.losses, pop.labels, 36, 4, REPS)
    _, _, se = summary(draws)
    assert 5.0 * float(se) < 0.01 * abs(float(pop.total))


def test_estimator_is_exact_at_a_census():
    """K_c = N_c leaves nothing to estimate."""
    pop = random_population(jax.random.key(2), 60, 12, scale=(1.0, 3.0))
    draws = stratified_draws(jax.random.key(3), pop.losses, pop.labels, 48, 12, 2000)
    assert jnp.allclose(draws, pop.total, rtol=0, atol=1e-9)
