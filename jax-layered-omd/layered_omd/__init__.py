"""Layered optimistic mirror descent for discounted Markov games (arXiv:2609.00504)."""

from .algorithm import run_full_feedback, run_partial_feedback
from .gap import cce_gap, general_policy_gap, per_episode_gap
from .games import MarkovGame, StationaryGame, benchmark_games, discounted_truncation
from .mirror import prox_step, step_sizes
from .values import best_response, player_values, uniform_policies, value

__all__ = [
    "MarkovGame",
    "StationaryGame",
    "benchmark_games",
    "discounted_truncation",
    "run_full_feedback",
    "run_partial_feedback",
    "prox_step",
    "step_sizes",
    "player_values",
    "best_response",
    "uniform_policies",
    "value",
    "cce_gap",
    "general_policy_gap",
    "per_episode_gap",
]
