"""DCTAS: cost-aware track-and-stop for Condorcet dueling bandits."""

from dctas.core import (
    ALPHA_EPS,
    C,
    DCTASResult,
    CostFractions,
    bernoulli_kl,
    beta,
    chernoff_stop,
    d,
    dctas,
    empirical_preference,
    glr,
    optimal_cost_fractions,
    plug_in_winner,
    weight_pull_allocation,
)

__all__ = [
    "ALPHA_EPS",
    "C",
    "CostFractions",
    "DCTASResult",
    "bernoulli_kl",
    "beta",
    "chernoff_stop",
    "d",
    "dctas",
    "empirical_preference",
    "glr",
    "optimal_cost_fractions",
    "plug_in_winner",
    "weight_pull_allocation",
]
