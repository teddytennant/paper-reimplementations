"""Packing-aware ternary weight routing and exact linear-operator compilation."""

from .compile import CompiledLinear, SharedSum, compile_routed_linear, estimate_saving
from .routing import (
    TernaryCandidates,
    assignment_weights,
    classify_groups,
    forward_weights,
    group_routing_loss,
    group_sensitivity,
    homogeneity_loss,
    protection_mask,
    soft_ternary_probs,
    ternary_quantize,
    training_objective,
)

__all__ = [
    "CompiledLinear",
    "SharedSum",
    "TernaryCandidates",
    "assignment_weights",
    "classify_groups",
    "compile_routed_linear",
    "estimate_saving",
    "forward_weights",
    "group_routing_loss",
    "group_sensitivity",
    "homogeneity_loss",
    "protection_mask",
    "soft_ternary_probs",
    "ternary_quantize",
    "training_objective",
]

__version__ = "0.1.0"
