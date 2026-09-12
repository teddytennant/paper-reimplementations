"""JAX implementation of DataShifts (arXiv:2609.11918)."""

from .algorithm import DataShiftsResult, datashifts
from .estimators import concept_shift, debiased_w_beta, plugin_w_beta
from .sinkhorn import cost_matrix, entropic_ot, sinkhorn

__all__ = [
    "DataShiftsResult",
    "concept_shift",
    "cost_matrix",
    "datashifts",
    "debiased_w_beta",
    "entropic_ot",
    "plugin_w_beta",
    "sinkhorn",
]

__version__ = "0.1.0"
