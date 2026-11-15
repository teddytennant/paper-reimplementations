"""Section 3 of the paper: algorithm-dependent learnability and its three
sufficiency results."""

from ugtl.theory.bounds import (
    alpha_greedy,
    alpha_local_search,
    alpha_pgd,
    greedy_value_oracle_ratio,
    pgd_step_size,
)
from ugtl.theory.convex import (
    BoxQuadratic,
    adversarial_gradient_error,
    projected_gradient_descent,
)
from ugtl.theory.pac_gap import HiddenJackpotCoverage
from ugtl.theory.submodular import (
    FacilityLocation,
    MaxCoverage,
    MaxCut,
    PerturbedOracle,
    PowerModular,
    brute_force_max,
    greedy,
    local_search,
    submodularity_ratio,
)

__all__ = [
    "alpha_greedy",
    "alpha_local_search",
    "alpha_pgd",
    "greedy_value_oracle_ratio",
    "pgd_step_size",
    "BoxQuadratic",
    "HiddenJackpotCoverage",
    "adversarial_gradient_error",
    "projected_gradient_descent",
    "FacilityLocation",
    "MaxCoverage",
    "MaxCut",
    "PerturbedOracle",
    "PowerModular",
    "brute_force_max",
    "greedy",
    "local_search",
    "submodularity_ratio",
]
