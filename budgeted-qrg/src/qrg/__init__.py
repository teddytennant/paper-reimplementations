"""Budgeted quotient-residual guidance (arXiv:2609.31222)."""

from qrg.kernel import (
    LOCAL,
    PREDNEXT,
    budgeted_delivery,
    budgeted_descent,
    combine_step,
    covector_lift,
    evaluation_state,
    horizontal_lift,
    product_budget_delivery,
    qrg_step,
    sampler_budget,
)
from qrg.quotients import pairwise_contact, pairwise_distance

__all__ = [
    "LOCAL",
    "PREDNEXT",
    "budgeted_delivery",
    "budgeted_descent",
    "combine_step",
    "covector_lift",
    "evaluation_state",
    "horizontal_lift",
    "pairwise_contact",
    "pairwise_distance",
    "product_budget_delivery",
    "qrg_step",
    "sampler_budget",
]
