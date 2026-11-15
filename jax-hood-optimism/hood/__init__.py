"""Higher-order optimism with discounting (HOOD), arXiv:2609.04113.

Abbadi, Laraki and Mertikopoulos, "Constant regret in general games via
higher-order optimism". Symbols and equation numbers throughout the package
refer to that paper.

Importing this package turns on float64 and forces highest matmul precision.
The (N+1)-th order predictor of eq. (22) is an alternating binomial sum, and in
float32 the cancellation between its terms eats most of the significant digits;
on a TF32 accelerator it eats the rest.
"""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from hood.algorithms import run_hood, run_optftrl  # noqa: E402
from hood.choice import (  # noqa: E402
    entropic_choice,
    lifted_choice,
    lifted_value,
    regularizer_constant,
    solve_lambda,
)
from hood.games import (  # noqa: E402
    NormalFormGame,
    coordination_game,
    matching_pennies,
    payoff_vectors,
    random_game,
)
from hood.predictors import discount_factor, discounted_difference, hood_predictor  # noqa: E402
from hood.regret import cce_gap, empirical_distribution, individual_regret, regret_bound  # noqa: E402

__all__ = [
    "NormalFormGame",
    "cce_gap",
    "coordination_game",
    "discount_factor",
    "discounted_difference",
    "empirical_distribution",
    "entropic_choice",
    "hood_predictor",
    "individual_regret",
    "lifted_choice",
    "lifted_value",
    "matching_pennies",
    "payoff_vectors",
    "random_game",
    "regret_bound",
    "regularizer_constant",
    "run_hood",
    "run_optftrl",
    "solve_lambda",
]
