"""JAX implementation of the Exact Log-Symplectic Flow (arXiv 2609.24737)."""

from ananke.block import (
    ctan_res,
    init_ctan_res,
    init_tiny_classifier,
    tiny_classifier,
    train_tiny_classifier,
)
from ananke.flow import (
    apply_dual_shear,
    apply_rotation,
    exact_circular_flow,
    flow_quantities,
    flow_vs_rk4_error,
    integrate_polar_rk4,
    product_flow,
    rotation_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "apply_dual_shear",
    "apply_rotation",
    "ctan_res",
    "exact_circular_flow",
    "flow_quantities",
    "flow_vs_rk4_error",
    "init_ctan_res",
    "init_tiny_classifier",
    "integrate_polar_rk4",
    "product_flow",
    "rotation_matrix",
    "tiny_classifier",
    "train_tiny_classifier",
]
