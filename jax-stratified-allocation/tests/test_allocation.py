"""Theorem 8 and Corollaries 9 and 10: where the optimum sits, and what the class
multiplier does to it."""

import jax
import jax.numpy as jnp
import pytest

from stratified_allocation import (
    allocation_efficiency_loss,
    class_counts,
    class_dispersions,
    equidispersed_population,
    equidispersed_relative_efficiency,
    neyman_allocation,
    proportional_allocation,
    random_population,
    variance_stratified,
    weighted_losses,
)


def _brute_force_optimum(counts, disp, budget):
    """The integer argmin of Equation (8) over every split with K_0 + K_1 = K."""
    k0 = jnp.arange(1, budget, dtype=jnp.float64)
    k0 = k0[(k0 <= float(counts[0])) & (budget - k0 <= float(counts[1]))]
    var = jax.vmap(lambda a: variance_stratified(counts, disp, jnp.array([a, budget - a])))(k0)
    best = int(jnp.argmin(var))
    return int(k0[best]), float(var[best])


@pytest.mark.parametrize(
    "n, n1, budget, scale",
    [
        (1000, 100, 100, (1.0, 2.0)),
        (1000, 100, 250, (1.0, 0.4)),
        (800, 200, 120, (1.0, 1.0)),
        (2000, 40, 50, (1.0, 3.0)),
    ],
)
def test_neyman_allocation_is_the_argmin(n, n1, budget, scale):
    """Equation (11) against a sweep of every feasible integer allocation."""
    pop = random_population(jax.random.key(21), n, n1, scale=scale, shift=(0.0, 1.0))
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)

    exact = neyman_allocation(counts, disp, budget)
    # Equation (11) is the solution of the problem with 0 < K_c <= N_c only when that
    # box is slack. These settings keep it slack; the boundary case has its own test.
    assert float(exact[1]) <= float(counts[1]) and float(exact[0]) <= float(counts[0])
    brute_k0, brute_var = _brute_force_optimum(counts, disp, budget)

    # Equation (8) is convex in K_0 once the allocation-free correction is dropped, so
    # the integer minimiser is one of the two neighbours of the continuous optimum.
    assert abs(brute_k0 - float(exact[0])) <= 1.0

    # Rounding Equation (11) to the nearest integer can land on the further of the two
    # neighbours when the continuous optimum sits near a half, so the claim is that it
    # lands within a hair of the sweep's minimum, not that it always matches it.
    rounded = neyman_allocation(counts, disp, budget, integer=True)
    at_rounded = float(variance_stratified(counts, disp, jnp.asarray(rounded, jnp.float64)))
    assert abs(rounded[0] - float(exact[0])) <= 1.0
    assert at_rounded <= brute_var * 1.001


@pytest.mark.parametrize("n, n1, budget", [(1000, 100, 100), (2000, 40, 300)])
def test_efficiency_loss_formula_matches_the_variance_ratio(n, n1, budget):
    """Equation (12): Var(a)/Var(a*) = sum_c (a_c*)^2 / a_c, and it never dips below 1."""
    pop = random_population(jax.random.key(22), n, n1, scale=(1.0, 2.0))
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    star = neyman_allocation(counts, disp, budget)

    for k0 in range(2, budget, max(1, budget // 17)):
        alloc = jnp.array([float(k0), float(budget - k0)])
        ratio = variance_stratified(counts, disp, alloc, fpc=False) / variance_stratified(
            counts, disp, star, fpc=False
        )
        formula = allocation_efficiency_loss(alloc / budget, star / budget)
        assert jnp.allclose(ratio, formula, rtol=1e-12, atol=0.0)
        assert float(formula) >= 1.0 - 1e-12


def test_optimal_ratio_is_sigma1_over_sigma0():
    """Equation (14): K_1*/K_0* = N_1 S_1 / (N_0 S_0) = sigma_1 / sigma_0."""
    for n, n1, ratio in [(1000, 50, 2.0), (1000, 200, 0.5), (500, 25, 3.0)]:
        pop = random_population(jax.random.key(23), n, n1, scale=(1.0, ratio))
        counts = class_counts(pop.labels)
        weighted = class_dispersions(pop.losses, pop.labels)
        unweighted = class_dispersions(pop.raw, pop.labels)
        star = neyman_allocation(counts, weighted, 100)
        assert jnp.allclose(star[1] / star[0], unweighted[1] / unweighted[0], rtol=1e-12)


def test_imbalance_ratio_cancels_from_the_optimum():
    """Corollary 9: once the loss carries omega_1 = N_0/N_1, the optimal split does not
    depend on pi at all.

    The same unweighted dispersions are placed in folds of very different prevalence.
    omega_1 changes by a factor of 40 across them; the optimal allocation ratio does not
    move.
    """
    target = 1.7
    ratios, multipliers = [], []
    for n, n1 in [(2000, 25), (2000, 200), (1000, 500)]:
        base = equidispersed_population(jax.random.key(24), n, n1, sigma=1.0)
        raw = jnp.where(base.labels == 1, base.raw * target, base.raw)
        losses = weighted_losses(raw, base.labels)
        counts = class_counts(base.labels)
        star = neyman_allocation(counts, class_dispersions(losses, base.labels), 100)
        ratios.append(float(star[1] / star[0]))
        multipliers.append(float(counts[0] / counts[1]))

    assert max(multipliers) / min(multipliers) > 30.0
    for r in ratios:
        assert abs(r - target) < 1e-10


def test_equal_allocation_is_optimal_exactly_under_equal_unweighted_dispersion():
    """Corollary 9's condition (13), sigma_1 = sigma_0, iff N_0 S_0 = N_1 S_1."""
    pop = equidispersed_population(jax.random.key(25), 1000, 100, sigma=1.0)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    assert jnp.allclose(counts[0] * disp[0], counts[1] * disp[1], rtol=1e-12)
    star = neyman_allocation(counts, disp, 200)
    assert jnp.allclose(star, jnp.array([100.0, 100.0]), rtol=1e-12)

    # And proportional allocation is not optimal here, by a lot.
    prop = jnp.array(proportional_allocation(counts, 200), jnp.float64)
    assert float(allocation_efficiency_loss(prop / 200.0, star / 200.0)) > 2.0


@pytest.mark.parametrize("n, n1, budget", [(1000, 100, 100), (1000, 250, 200), (2000, 100, 400)])
def test_corollary_10_relative_efficiency(n, n1, budget):
    """Equation (15): Var_prop / Var_bal = n^2 / (4 N_0 N_1) = 1 / {4 pi (1 - pi)}.

    Stated under (13) and ignoring the finite-population corrections, so the population
    is built equidispersed before weighting and the variances are the fpc-free ones.
    """
    pop = equidispersed_population(jax.random.key(26), n, n1, sigma=1.3)
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    pi = n1 / n

    prop = jnp.array([budget * counts[0] / n, budget * counts[1] / n])
    bal = jnp.array([budget / 2.0, budget / 2.0])
    ratio = variance_stratified(counts, disp, prop, fpc=False) / variance_stratified(
        counts, disp, bal, fpc=False
    )
    assert jnp.allclose(ratio, equidispersed_relative_efficiency(pi), rtol=1e-11, atol=0.0)


def test_corollary_10_reproduces_the_papers_numbers():
    """The factors quoted after Equation (15) at the paper's three fold prevalences."""
    got = [float(equidispersed_relative_efficiency(p)) for p in (0.0037, 0.0071, 0.0143)]
    assert [round(v, 1) for v in got] == [67.8, 35.5, 17.7]


def test_optimum_sits_on_the_boundary_when_equation_11_is_infeasible():
    """Theorem 8 constrains 0 < K_c <= N_c, and at a large budget with few positives
    Equation (11) asks for more positives than the fold has.

    The constrained optimum is then a census of the positive stratum, K_1 = N_1, which
    is also what the truncation of Section 6.4 does.
    """
    pop = random_population(jax.random.key(21), 2000, 40, scale=(1.0, 3.0), shift=(0.0, 1.0))
    counts = class_counts(pop.labels)
    disp = class_dispersions(pop.losses, pop.labels)
    budget = 300
    assert float(neyman_allocation(counts, disp, budget)[1]) > float(counts[1])
    brute_k0, _ = _brute_force_optimum(counts, disp, budget)
    assert brute_k0 == budget - int(counts[1])
    assert neyman_allocation(counts, disp, budget, integer=True) == (260, 40)
