"""JAX kernel of rotation-based online robust kernel PCA (arXiv:2609.15488)."""

from jax_rkpca.kernel import empirical_kernel_map, linear_kernel, rbf_kernel
from jax_rkpca.distance import half_squared_distance, loadings
from jax_rkpca.rotation import (
    apply_plane_rotation,
    rotation_matrix,
    type_i_vectors,
    type_ii_index,
    type_ii_vector,
)
from jax_rkpca.robust import psi, psi_dot, rotation_fraction
from jax_rkpca.update import RKPCState, init_state, orthonormalize, step, track
from jax_rkpca.metrics import (
    cosine_similarity_matched,
    generate_synthetic_stream,
    grassmann_distance,
    principal_angles,
    variance_weighted_grassmann,
)

__version__ = "0.1.0"

__all__ = [
    "RKPCState",
    "apply_plane_rotation",
    "cosine_similarity_matched",
    "empirical_kernel_map",
    "generate_synthetic_stream",
    "grassmann_distance",
    "half_squared_distance",
    "init_state",
    "linear_kernel",
    "loadings",
    "orthonormalize",
    "principal_angles",
    "psi",
    "psi_dot",
    "rbf_kernel",
    "rotation_fraction",
    "rotation_matrix",
    "step",
    "track",
    "type_i_vectors",
    "type_ii_index",
    "type_ii_vector",
    "variance_weighted_grassmann",
]
