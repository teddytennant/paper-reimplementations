"""JAX implementation of SS-ESOAP (arXiv:2608.29448)."""

from ._core import (
    Config,
    gamma_from_rho,
    kron_quadratic_form,
    offdiag_ratio,
    secant_scale,
)
from .optimizer import SSESOAPState, scale_by_ss_esoap, soap, ss_esoap

__all__ = [
    "Config",
    "SSESOAPState",
    "gamma_from_rho",
    "kron_quadratic_form",
    "offdiag_ratio",
    "scale_by_ss_esoap",
    "secant_scale",
    "soap",
    "ss_esoap",
]
