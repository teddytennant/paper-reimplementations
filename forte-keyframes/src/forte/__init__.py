"""FORTE keyframe selection: GP relevance scoring and exact coverage optimization."""

import jax

jax.config.update("jax_enable_x64", True)

from forte.coverage import coverage, marginal_gain, marginal_gains
from forte.gp import GPFit, fit_gp, minmax_normalize, predict
from forte.select import (
    KeyframeResult,
    ScoreResult,
    adaptive_score,
    exact_keyframes,
    select_keyframes,
)

__all__ = [
    "GPFit",
    "KeyframeResult",
    "ScoreResult",
    "adaptive_score",
    "coverage",
    "exact_keyframes",
    "fit_gp",
    "marginal_gain",
    "marginal_gains",
    "minmax_normalize",
    "predict",
    "select_keyframes",
]

__version__ = "0.1.0"
