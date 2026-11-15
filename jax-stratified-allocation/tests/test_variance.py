"""Propositions 5 and 6: the exact finite-population variance, and what it costs to
draw uniformly instead of stratifying."""

import jax
import jax.numpy as jnp
import pytest

from mc_util import (
    cached_stratified,
    cached_uniform,
    strat_pop,
    summary,
    unif_pop,
    variance_relative_se,
)
from stratified_allocation import (
    balanced_allocation,
    between_class_term,
    class_counts,
    class_dispersions,
    pooled_dispersion,
    proportional_allocation,
    random_population,
    uniform_minus_proportional,
    variance_stratified,
    variance_uniform,
)

REPS = 200_000

# (n, N_1, K). The middle one truncates: balanced allocation asks for 30 positives and
# the fold only has 10, so K_1 = N_1 and the positive stratum is a census.
SETTINGS = [(400, 40, 40), (500, 10, 60), (1000, 200, 100)]


def _check(draws, analytic):
    mc_var = float(summary(draws)[1])
    rel_se = float(variance_relative_se(draws))
    # The test only means something if the Monte Carlo error is small.
    assert rel_se < 0.02, f"replicate count too low, relative SE {rel_se:.3f}"
    assert abs(mc_var / float(analytic) - 1.0) < 6.0 * rel_se


@pytest.mark.parametrize("n, n1, budget", SETTINGS)
@pytest.mark.parametrize("rule", ["proportional", "balanced"])
def test_exact_variance_matches_monte_carlo(n, n1, budget, rule):
    """Equation (8) against the sampling variance of the same estimator."""
    pop = strat_pop(n, n1)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    alloc = (
        proportional_allocation(counts, budget)
        if rule == "proportional"
        else balanced_allocation(counts, budget)
    )
    draws = cached_stratified(f"s{n}-{n1}", 12, pop.losses, pop.labels, *alloc, REPS)
    _check(draws, variance_stratified(counts, disp, jnp.asarray(alloc, jnp.float64)))


def test_the_truncating_setting_really_truncates():
    """Guard on the parametrisation above: one setting has to hit K_1 = N_1."""
    counts = class_counts(strat_pop(500, 10).labels)
    assert balanced_allocation(counts, 60) == (30, 10)
    assert int(counts[1]) == 10


@pytest.mark.parametrize("n, n1, budget", SETTINGS)
def test_uniform_variance_matches_monte_carlo(n, n1, budget):
    """Proposition 5's uniform case, n^2 (1 - K/n) S^2 / K."""
    pop = unif_pop(n, n1)
    draws = cached_uniform(f"u{n}-{n1}", 11, pop.losses, budget, REPS)
    _check(draws, variance_uniform(n, pooled_dispersion(pop.losses), budget))


def test_census_has_zero_variance():
    pop = strat_pop(400, 40)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    v = variance_stratified(counts, disp, jnp.array([360.0, 40.0]))
    assert abs(float(v)) < 1e-6 * float(jnp.sum(counts * disp**2))


@pytest.mark.parametrize("n, n1, budget", [(500, 50, 50), (1000, 100, 200), (600, 60, 60)])
def test_proposition_6_identity_is_exact(n, n1, budget):
    """Equation (9), as an identity and not an inequality.

    Var_unif - Var_prop is computed two ways: once by subtracting the two closed forms
    of Proposition 5, once from the right-hand side of (9). The settings are chosen so
    that K N_c / n is an integer for both classes, which is what (9) assumes.
    """
    pop = random_population(jax.random.key(7), n, n1, scale=(1.0, 2.5), shift=(0.0, 2.0))
    counts = class_counts(pop.labels)
    alloc = jnp.array([budget * float(counts[0]) / n, budget * float(counts[1]) / n])
    assert jnp.allclose(alloc, jnp.round(alloc))

    disp = class_dispersions(pop.losses, pop.labels)
    direct = variance_uniform(n, pooled_dispersion(pop.losses), budget) - variance_stratified(
        counts, disp, alloc
    )
    assert jnp.allclose(direct, uniform_minus_proportional(pop, budget), rtol=1e-11, atol=0.0)


def test_proposition_6_is_positive_in_the_rare_event_regime():
    """The stated sufficient condition, B >= sum_c S_c^2, and its consequence.

    Under Equation (6) the class means of l differ by a factor of order omega_1, so B is
    enormous next to the dispersions and uniform sampling loses by a wide margin.
    """
    pop = random_population(jax.random.key(8), 1000, 20, scale=(1.0, 1.0), shift=(1.0, 1.0))
    b = between_class_term(pop.losses, pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    assert float(b) > float(jnp.sum(disp**2))
    assert float(uniform_minus_proportional(pop, 100)) > 0.0


def test_uniform_beats_stratified_nowhere_when_the_condition_holds():
    """Both sides of Proposition 6 agree on the sign at every budget we try."""
    pop = random_population(jax.random.key(9), 500, 50, scale=(1.0, 2.0), shift=(0.0, 1.0))
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    for budget in (50, 100, 150, 200, 250):
        alloc = jnp.array([budget * float(counts[0]) / 500.0, budget * float(counts[1]) / 500.0])
        gap = variance_uniform(500, pooled_dispersion(pop.losses), budget) - variance_stratified(
            counts, disp, alloc
        )
        assert float(gap) > 0.0
        assert jnp.allclose(gap, uniform_minus_proportional(pop, budget), rtol=1e-11, atol=0.0)
