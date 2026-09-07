"""Exact finite-population variances and the optimal allocation, Sections 6.2 to 6.4.

Every function here is a transcription of one displayed equation of arXiv:2609.04420.
None of them touch the data twice or go through a matmul, because Proposition 6 is a
difference of two nearly equal quantities and Corollary 10 is a ratio of them.
"""

import jax
import jax.numpy as jnp

from stratified_allocation.population import (
    Population,
    between_class_term,
    class_counts,
    class_dispersions,
    pooled_dispersion,
)


def variance_stratified(
    counts: jax.Array, dispersions: jax.Array, allocation, fpc: bool = True
) -> jax.Array:
    """Equation (8): Var(Lhat) = sum_c (N_c^2 / K_c)(1 - K_c/N_c) S_c^2.

    The correction contributes -sum_c N_c S_c^2, which does not depend on the
    allocation, so dropping it (``fpc=False``) leaves the argmin of Theorem 8 unchanged
    while changing the ratios of Corollaries 10 and 12. Corollary 10 is stated "ignoring
    the finite-population corrections", so that is the setting to use with it.
    """
    counts = jnp.asarray(counts, jnp.float64)
    dispersions = jnp.asarray(dispersions, jnp.float64)
    alloc = jnp.asarray(allocation, jnp.float64)
    main = counts**2 * dispersions**2 / alloc
    if not fpc:
        return jnp.sum(main)
    return jnp.sum(main - counts * dispersions**2)


def variance_uniform(n: int, pooled: jax.Array, budget: int, fpc: bool = True) -> jax.Array:
    """Proposition 5, uniform design: Var(Lhat) = n^2 (1 - K/n) S^2 / K."""
    n = float(n)
    main = n**2 * pooled**2 / budget
    if not fpc:
        return main
    return main * (1.0 - budget / n)


def variance_stratified_population(
    population: Population, allocation, fpc: bool = True
) -> jax.Array:
    """Equation (8) evaluated directly on a :class:`Population`."""
    counts = class_counts(population.labels)
    disp = class_dispersions(population.losses, population.labels)
    return variance_stratified(counts, disp, allocation, fpc=fpc)


def variance_uniform_population(population: Population, budget: int, fpc: bool = True) -> jax.Array:
    """Proposition 5's uniform case evaluated directly on a :class:`Population`."""
    return variance_uniform(
        population.n, pooled_dispersion(population.losses), budget, fpc=fpc
    )


def uniform_minus_proportional(population: Population, budget: int) -> jax.Array:
    """Equation (9), the right-hand side of Proposition 6.

        Var_unif - Var_prop = n^2 (1 - K/n) / (K (n-1))
                              * [ B + (1/n) sum_c N_c S_c^2 - sum_c S_c^2 ],

    with B = sum_c N_c (lbar_c - lbar)^2 and K_c = K N_c / n on the stratified side. The
    identity holds exactly only when K N_c / n is an integer for both classes, so pick
    n, K and N_1 accordingly when you want to check it.
    """
    n = float(population.n)
    counts = class_counts(population.labels)
    disp = class_dispersions(population.losses, population.labels)
    b = between_class_term(population.losses, population.labels)
    bracket = b + jnp.sum(counts * disp**2) / n - jnp.sum(disp**2)
    return n**2 * (1.0 - budget / n) / (budget * (n - 1.0)) * bracket


def allocation_efficiency_loss(shares, optimal_shares) -> jax.Array:
    """Equation (12): Var(a)/Var(a*) = sum_c (a_c*)^2 / a_c, at least 1.

    Shares are a_c = K_c / K. This is the finite-population-correction-free ratio, which
    is what Theorem 8 compares, since the correction term is allocation-free.
    """
    a = jnp.asarray(shares, jnp.float64)
    astar = jnp.asarray(optimal_shares, jnp.float64)
    return jnp.sum(astar**2 / a)


def equidispersed_relative_efficiency(prevalence) -> jax.Array:
    """Equation (15) of Corollary 10: Var_prop / Var_bal = 1 / {4 pi (1 - pi)}.

    Holds under condition (13), sigma_1 = sigma_0, ignoring the finite-population
    corrections, at a common budget K with K_0 = K_1 = K/2.
    """
    pi = jnp.asarray(prevalence, jnp.float64)
    return 1.0 / (4.0 * pi * (1.0 - pi))


def realised_allocation_ratio(prevalence, retention) -> jax.Array:
    """Equation (17) of Corollary 12: gamma = K_1/K_0 = min{2 pi / f, 1}.

    With K* = ceil(f n) and N_1 = pi n, the truncation K_c = min{N_c, floor(K*/2)} gives
    K_0 = floor(K*/2) and K_1 = min(N_1, floor(K*/2)), so the ratio depends on pi and f
    alone, not on the fold size n.
    """
    pi = jnp.asarray(prevalence, jnp.float64)
    f = jnp.asarray(retention, jnp.float64)
    return jnp.minimum(2.0 * pi / f, 1.0)


def efficiency_prediction(prevalence, retention) -> jax.Array:
    """Equation (18) of Corollary 12: A(pi, f) = gamma / {pi (1 - pi) (1 + gamma)^2}.

    The predicted efficiency of the truncated class-balanced design relative to
    proportional allocation, at the same realised total K_0 + K_1, under condition (13)
    and ignoring the finite-population corrections. It is Corollary 10's factor
    1/{4 pi (1 - pi)} divided by the truncation's own loss (1 + gamma)^2 / (4 gamma)
    from Equation (12). No free parameter and no fitted quantity, which is the point of
    Table 2.
    """
    gamma = realised_allocation_ratio(prevalence, retention)
    pi = jnp.asarray(prevalence, jnp.float64)
    return gamma / (pi * (1.0 - pi) * (1.0 + gamma) ** 2)
