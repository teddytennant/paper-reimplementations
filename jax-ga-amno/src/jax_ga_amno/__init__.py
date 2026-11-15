"""GA-AMNO allocation and gauge-aware transport (arXiv:2609.15620)."""

from jax_ga_amno.allocation import (
    allocate,
    displacements,
    importance_weights,
    physical_indicators,
    project_domain,
)
from jax_ga_amno.geometry import (
    edge_descriptors,
    four_connected_neighbors,
    knn_neighbors,
    local_jacobians,
    reference_grid,
)
from jax_ga_amno.model import GAAMNOConfig, ga_amno_forward, init_ga_amno
from jax_ga_amno.transport import (
    apply_transport,
    attention_aggregate,
    gauge_layer,
    residual_update,
    transport_matrix,
)

__all__ = [
    "GAAMNOConfig",
    "allocate",
    "apply_transport",
    "attention_aggregate",
    "displacements",
    "edge_descriptors",
    "four_connected_neighbors",
    "ga_amno_forward",
    "gauge_layer",
    "importance_weights",
    "init_ga_amno",
    "knn_neighbors",
    "local_jacobians",
    "physical_indicators",
    "project_domain",
    "reference_grid",
    "residual_update",
    "transport_matrix",
]
