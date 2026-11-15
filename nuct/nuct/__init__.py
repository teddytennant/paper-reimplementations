"""Classical cores of the nonuniform Chebyshev transform (arXiv:2609.40277).

Quantum circuits, block encodings, and gate counts are out of scope. What
is here is the classical skeleton those circuits are built on:

* Definition 1.1 and Eq. (3): equispaced nodes, unnormalized coefficients,
  normalized matrix ``C_N``.
* Theorem 5.1: ``C_N = (F_t + F_{t^-}) / 2``.
* Eq. (6): integer split of a fixed-point node, and the phase identity.
* Bessel-Chebyshev truncation of ``H(x, z) = exp(-i π x z / 2)``.
* Eq. (31): ``F_τ^{(K)} = sum_r diag(v_r) F M_σ diag(u_r)``.
* Angle-output rounding with no ``κ``, and the node-error operator bound.
* The classical row oracle for ``M_σ``.

Index convention: row = frequency / degree, column = node.
Tie convention for ``round``: half away from zero. See ``nuct.split``.
"""

from __future__ import annotations

from nuct._config import ensure_x64

ensure_x64()

from nuct.bounds import (
    arccos_kappa,
    collision_hypothesis_holds,
    kappa_scaled_input_error,
    minimum_node_bits,
    node_error_operator_bound,
    nudft_operator_distance,
    output_rounding_chebyshev_error,
    realized_input_angle_error,
    round_arccos_output,
    worst_case_angle_chebyshev_error,
)
from nuct.chebyshev import (
    chebyshev_T,
    chebyshev_matrix,
    coefficients_from_matrix,
    equispaced_nodes,
    nuct_from_nufft,
    nuct_order,
    nuct_times,
    nudft_matrix,
    reflected_times,
    theorem_51_nodes,
    unitary_dft,
    unnormalized_coefficients,
)
from nuct.kernel import (
    bessel_coeff_table,
    embedding_matrix,
    factorized_nudft,
    frequency_coordinates,
    frequency_phase_rescaling,
    frequency_weights,
    interaction_kernel,
    node_amplitudes,
    truncated_kernel,
)
from nuct.oracle import (
    MULTIPLICITY_CAP,
    embedding_spectral_norm,
    max_multiplicity,
    multiplicities,
    numerical_embedding_spectral_norm,
    row_support,
)
from nuct.split import (
    TIE_CONVENTION,
    NodeSplit,
    quantize_unit_interval,
    split_fixed_point,
    split_phase_sides,
    store_nuct_nodes,
)

__all__ = [
    "MULTIPLICITY_CAP",
    "TIE_CONVENTION",
    "NodeSplit",
    "arccos_kappa",
    "bessel_coeff_table",
    "chebyshev_T",
    "chebyshev_matrix",
    "coefficients_from_matrix",
    "collision_hypothesis_holds",
    "embedding_matrix",
    "embedding_spectral_norm",
    "equispaced_nodes",
    "factorized_nudft",
    "frequency_coordinates",
    "frequency_phase_rescaling",
    "frequency_weights",
    "interaction_kernel",
    "kappa_scaled_input_error",
    "max_multiplicity",
    "minimum_node_bits",
    "multiplicities",
    "node_amplitudes",
    "node_error_operator_bound",
    "nuct_from_nufft",
    "nuct_order",
    "nuct_times",
    "nudft_matrix",
    "nudft_operator_distance",
    "numerical_embedding_spectral_norm",
    "output_rounding_chebyshev_error",
    "quantize_unit_interval",
    "realized_input_angle_error",
    "reflected_times",
    "round_arccos_output",
    "row_support",
    "split_fixed_point",
    "split_phase_sides",
    "store_nuct_nodes",
    "theorem_51_nodes",
    "truncated_kernel",
    "unitary_dft",
    "unnormalized_coefficients",
    "worst_case_angle_chebyshev_error",
]
