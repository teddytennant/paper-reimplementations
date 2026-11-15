"""FRAC: fractional state-space mixer kernel (arXiv:2609.36314)."""

from . import _host

_host.preload()

from .bank import (
    geometric_timescales,
    implemented_rho_beta,
    mittag_leffler_e_alpha,
    soe_coeffs_spectral,
    soe_eval,
    zoh_closed_form,
    zoh_rho_beta,
)
from .layer import (
    apply_layer,
    apply_layer_step,
    decode_autoregressive,
    empty_cache,
    init_frac_layer,
)
from .transition import (
    affine_compose,
    frac_step,
    parallel_scan,
    sequential_scan,
    transition_sequence,
)

__all__ = [
    "affine_compose",
    "apply_layer",
    "apply_layer_step",
    "decode_autoregressive",
    "empty_cache",
    "frac_step",
    "geometric_timescales",
    "implemented_rho_beta",
    "init_frac_layer",
    "mittag_leffler_e_alpha",
    "parallel_scan",
    "sequential_scan",
    "soe_coeffs_spectral",
    "soe_eval",
    "transition_sequence",
    "zoh_closed_form",
    "zoh_rho_beta",
]
