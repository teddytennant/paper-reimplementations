"""Corollary 12: the realised ratio gamma and the parameter-free prediction A(pi, f)."""

import jax
import jax.numpy as jnp
import pytest

from mc_util import stratified_draws, summary, variance_relative_se
from stratified_allocation import (
    balanced_allocation,
    class_counts,
    class_dispersions,
    efficiency_prediction,
    equidispersed_population,
    realised_allocation_ratio,
    variance_stratified,
)

# The paper's Table 2: rows are the mean fold prevalences at h = 5, 10, 20, columns are
# the retention grid f, entries are A(pi, f) with gamma in parentheses.
TABLE_2 = {
    0.0037: [(0.01, 66.3, 0.74), (0.025, 47.8, 0.30), (0.05, 30.5, 0.15), (0.10, 17.4, 0.07), (0.20, 9.3, 0.04)],
    0.0071: [(0.01, 35.5, 1.00), (0.025, 32.8, 0.57), (0.05, 24.4, 0.28), (0.10, 15.4, 0.14), (0.20, 8.8, 0.07)],
    0.0143: [(0.01, 17.7, 1.00), (0.025, 17.7, 1.00), (0.05, 16.4, 0.57), (0.10, 12.3, 0.29), (0.20, 7.8, 0.14)],
}


def test_table_2_is_reproduced():
    """Every cell of Table 2, from Equations (17) and (18) alone."""
    for pi, row in TABLE_2.items():
        for f, a, gamma in row:
            assert round(float(efficiency_prediction(pi, f)), 1) == a
            assert round(float(realised_allocation_ratio(pi, f)), 2) == gamma


def _designs(n, n1, retention):
    """Realised allocations for the truncated balanced rule and its proportional
    comparison at the same realised total K_0 + K_1."""
    budget = int(jnp.ceil(retention * n))
    counts = class_counts(jnp.concatenate([jnp.zeros(n - n1, jnp.int32), jnp.ones(n1, jnp.int32)]))
    bal = balanced_allocation(counts, budget)
    total = bal[0] + bal[1]
    prop = jnp.array([total * counts[0] / n, total * counts[1] / n])
    return jnp.asarray(bal, jnp.float64), prop


@pytest.mark.parametrize(
    "n, n1, retention",
    [(1000, 50, 0.4), (4000, 40, 0.05), (2000, 40, 0.10), (4000, 1000, 0.02), (2000, 200, 0.05)],
)
def test_A_is_the_fpc_free_variance_ratio(n, n1, retention):
    """Equation (18) as an exact identity.

    A(pi, f) is the ratio of proportional to truncated-balanced variance, ignoring the
    finite-population corrections, under condition (13), with proportional allocation
    given the same realised total K_0 + K_1 that the truncation actually spends. That
    last part matters: the truncated design draws fewer than K rows when positives are
    scarce, and comparing it against a design that spends the full K would not be the
    comparison Equation (12) makes.
    """
    pop = equidispersed_population(jax.random.key(31), n, n1, sigma=1.1)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    bal, prop = _designs(n, n1, retention)

    assert jnp.allclose(bal[1] / bal[0], realised_allocation_ratio(n1 / n, retention), rtol=1e-12)
    ratio = variance_stratified(counts, disp, prop, fpc=False) / variance_stratified(
        counts, disp, bal, fpc=False
    )
    assert jnp.allclose(ratio, efficiency_prediction(n1 / n, retention), rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("n, n1, retention", [(4000, 1000, 0.02), (2000, 500, 0.02)])
def test_A_matches_simulated_variances(n, n1, retention):
    """The prediction against Monte Carlo variances of the two implemented designs.

    Design points where the truncation is slack (gamma = 1) and both sampling fractions
    are small, so the finite-population corrections that Equation (18) drops are worth
    under a percent. Where the truncation binds they are not, which the next test says
    out loud.
    """
    reps = 200_000
    pop = equidispersed_population(jax.random.key(32), n, n1, sigma=1.1)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    bal, prop = _designs(n, n1, retention)
    prop_int = (int(round(float(prop[0]))), int(round(float(prop[1]))))

    mc = {}
    for name, alloc in [("bal", (int(bal[0]), int(bal[1]))), ("prop", prop_int)]:
        draws = stratified_draws(jax.random.key(33), pop.losses, pop.labels, *alloc, reps)
        var = float(summary(draws)[1])
        rel = float(variance_relative_se(draws))
        assert rel < 0.02
        # Each design's Monte Carlo variance agrees with its own closed form first.
        exact = float(variance_stratified(counts, disp, jnp.asarray(alloc, jnp.float64)))
        assert abs(var / exact - 1.0) < 6.0 * rel
        mc[name] = var

    predicted = float(efficiency_prediction(n1 / n, retention))
    assert abs(mc["prop"] / mc["bal"] / predicted - 1.0) < 0.05


@pytest.mark.parametrize("n, n1, retention", [(2000, 40, 0.10), (4000, 40, 0.05)])
def test_A_understates_the_gain_where_the_truncation_binds(n, n1, retention):
    """gamma < 1 means K_1 = N_1, which is a census of the positive stratum.

    Equation (18) drops the finite-population corrections, so it does not see that the
    truncated design's positive stratum contributes no variance at all. At these points
    the exact ratio is several times A, and the exact variance is the one the simulation
    reproduces.
    """
    pop = equidispersed_population(jax.random.key(34), n, n1, sigma=1.1)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    bal, prop = _designs(n, n1, retention)
    assert float(bal[1]) == float(counts[1])
    assert float(realised_allocation_ratio(n1 / n, retention)) < 1.0

    exact_ratio = float(
        variance_stratified(counts, disp, prop) / variance_stratified(counts, disp, bal)
    )
    assert exact_ratio > float(efficiency_prediction(n1 / n, retention))

    draws = stratified_draws(
        jax.random.key(35), pop.losses, pop.labels, int(bal[0]), int(bal[1]), 60_000
    )
    var = float(summary(draws)[1])
    rel = float(variance_relative_se(draws))
    assert abs(var / float(variance_stratified(counts, disp, bal)) - 1.0) < 6.0 * rel


def test_P2_A_decreases_in_prevalence_at_fixed_retention():
    """Prediction P2 of Section 6.4."""
    pis = jnp.linspace(0.001, 0.45, 400)
    for f in (0.01, 0.025, 0.05, 0.10, 0.20, 0.5, 1.0):
        a = efficiency_prediction(pis, f)
        assert bool(jnp.all(jnp.diff(a) < 0.0))


def test_P3_A_is_nonincreasing_in_retention_and_strict_past_2pi():
    """Prediction P3 of Section 6.4."""
    fs = jnp.linspace(0.002, 1.0, 500)
    for pi in (0.0037, 0.0071, 0.0143, 0.05, 0.2):
        a = efficiency_prediction(pi, fs)
        d = jnp.diff(a)
        assert bool(jnp.all(d <= 1e-15))
        strict = fs[:-1] > 2.0 * pi
        assert bool(jnp.all(jnp.where(strict, d < 0.0, True)))
        # Below f = 2 pi the truncation does not bind and A does not depend on f.
        flat = fs < 2.0 * pi
        if bool(jnp.any(flat)):
            assert jnp.allclose(a[flat], a[flat][0], rtol=1e-12)


def test_gamma_does_not_depend_on_fold_size():
    """Equation (17) is free of n, and the implemented rule agrees."""
    for n in (1000, 4000, 20000):
        n1 = int(0.02 * n)
        bal, _ = _designs(n, n1, 0.10)
        assert jnp.allclose(bal[1] / bal[0], realised_allocation_ratio(0.02, 0.10), rtol=1e-12)
