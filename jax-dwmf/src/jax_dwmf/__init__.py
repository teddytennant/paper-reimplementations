"""Diagonally weighted matrix-function dequantization (arXiv:2609.10729).

Implements Algorithm 1 / Theorem 16: sample heavy coordinates of D
(Lemma 15), reduce f(γI + DAD) to a principal block, and store a
support-sparse classical representation N with operator-norm guarantees.

The matrix function on the retained block is evaluated exactly by dense
eigh (not QSVT polynomial approximation).
"""

from .algorithm import (
    SparseMatrixFunction,
    diagonal_truncation_transform,
    materialize_N,
    query_N,
)
from .random_features import (
    cosine_kernel_entry,
    estimate_kernel_entry,
    optimized_feature_weights,
    sample_optimized_features,
    shift_invariant_kernel,
)
from .truncation import (
    frobenius_norm_diag,
    heavy_set,
    sample_count_R,
    sample_index,
    truncate_diagonal,
)

__all__ = [
    "SparseMatrixFunction",
    "cosine_kernel_entry",
    "diagonal_truncation_transform",
    "estimate_kernel_entry",
    "frobenius_norm_diag",
    "heavy_set",
    "materialize_N",
    "optimized_feature_weights",
    "query_N",
    "sample_count_R",
    "sample_index",
    "sample_optimized_features",
    "shift_invariant_kernel",
    "truncate_diagonal",
]

__version__ = "0.1.0"
