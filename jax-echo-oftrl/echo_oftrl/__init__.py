"""ECHO-OFTRL, from "Constant Individual Regret in General Games" (arXiv:2608.31166).

float64 is switched on at import. The lifted map Q_m of Lemma 3.1 is found by
bisecting a function containing 1/sqrt(lam) and 1/sqrt(1-lam), whose root sits
within 1e-14 of 1 once a player commits to an action, and regret is a
difference of accumulated payoffs. Neither survives float32, let alone the TF32
matmuls JAX uses for float32 on NVIDIA.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .cascade import (  # noqa: E402
    apply_A,
    apply_delta,
    apply_I_minus_delta,
    ema_cascade_predict,
    ema_cascade_step,
    lag,
    run_cascade,
)
from .dynamics import (  # noqa: E402
    DEFAULT_ETA,
    Trajectory,
    cascade_params,
    replay_from_gradients,
    run_game,
)
from .games import (  # noqa: E402
    biased_matching_pennies,
    duality_gap,
    matching_pennies,
    payoff_vectors,
    random_game,
    rock_paper_scissors,
    shifted_gradients,
)
from .lifted import (  # noqa: E402
    entropy,
    gamma_const,
    lifted_argmax,
    psi_tilde,
    stationarity,
    theoretical_learning_rate,
    xi_const,
)

__all__ = [
    "DEFAULT_ETA",
    "Trajectory",
    "apply_A",
    "apply_I_minus_delta",
    "apply_delta",
    "biased_matching_pennies",
    "cascade_params",
    "duality_gap",
    "ema_cascade_predict",
    "ema_cascade_step",
    "entropy",
    "gamma_const",
    "lag",
    "lifted_argmax",
    "matching_pennies",
    "payoff_vectors",
    "psi_tilde",
    "random_game",
    "replay_from_gradients",
    "rock_paper_scissors",
    "run_cascade",
    "run_game",
    "shifted_gradients",
    "stationarity",
    "theoretical_learning_rate",
    "xi_const",
]
