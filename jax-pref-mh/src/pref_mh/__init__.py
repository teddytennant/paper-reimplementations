"""Pref-MH: exact Metropolis-Hastings from binary pairwise preference votes.

Implements the sampler of "When Metropolis and Hastings Meet Bradley and
Terry: Exact MCMC From Preference Voting" (arXiv:2609.00905). The one thing to
know: under the Bradley-Terry model the MH unnormalized density ratio equals
the preference odds (Eq. 6), and Eq. (8) turns N binary votes into an accept
probability whose average satisfies detailed balance exactly for every finite
N (Theorem 2), without ever estimating the odds.
"""

from .acceptance import (
    bt_log_odds,
    bt_win_prob,
    log_alpha_bt_mh,
    log_alpha_n_vote,
    log_alpha_plugin,
    marginal_acceptance,
    marginal_acceptance_multi,
)
from .finite import (
    DiscreteModel,
    oracle_mh_kernel,
    plugin_kernel,
    pref_mh_kernel,
    sampler_ops,
    stationary,
    total_variation,
)
from .judges import bradley_terry_judge, count_votes
from .sampler import oracle_mh_chain, plugin_mh_chain, pref_mh_chain
from .x64 import enable as enable_x64

__all__ = [
    "DiscreteModel",
    "bradley_terry_judge",
    "bt_log_odds",
    "bt_win_prob",
    "count_votes",
    "enable_x64",
    "log_alpha_bt_mh",
    "log_alpha_n_vote",
    "log_alpha_plugin",
    "marginal_acceptance",
    "marginal_acceptance_multi",
    "oracle_mh_chain",
    "oracle_mh_kernel",
    "plugin_kernel",
    "plugin_mh_chain",
    "pref_mh_chain",
    "pref_mh_kernel",
    "sampler_ops",
    "stationary",
    "total_variation",
]
