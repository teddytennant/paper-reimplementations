"""JAX implementation of arXiv:2609.11168 (MSTP-SVD / MRSTP-SVD)."""

from .decompose import (
    Factor,
    mrstp_svd,
    mstp_reconstruct,
    mstp_svd,
    relative_frobenius_error,
    stp_svd_matrix,
)
from .product import (
    frobenius_norm,
    identity_tensor,
    matrix_stp,
    t_product,
    tensor_kronecker,
    tensor_stp,
    tensor_transpose,
)
from .rearrange import kronecker_terms, rearrange
from .transforms import DFT, FFT, Identity, Transform, get_transform

__all__ = [
    "DFT",
    "FFT",
    "Factor",
    "Identity",
    "Transform",
    "frobenius_norm",
    "get_transform",
    "identity_tensor",
    "kronecker_terms",
    "matrix_stp",
    "mrstp_svd",
    "mstp_reconstruct",
    "mstp_svd",
    "rearrange",
    "relative_frobenius_error",
    "stp_svd_matrix",
    "t_product",
    "tensor_kronecker",
    "tensor_stp",
    "tensor_transpose",
]

__version__ = "0.1.0"
