"""IRLS for single-commodity fixed-charge network flow (arXiv:2609.09295)."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from jax_fcnfp_irls.irls import (
    IRLSResult,
    accept_step,
    continuation_update,
    flow_gap,
    initialize_smoothing,
    irls,
    refit_support,
)
from jax_fcnfp_irls.network import conservation_residual, node_arc_incidence
from jax_fcnfp_irls.ssn import (
    dual_ssn,
    free_arc_mask,
    newton_diag,
    newton_matrix,
    predictor_y,
    primal_from_dual,
)
from jax_fcnfp_irls.surrogate import (
    admissible_mu,
    irls_weights,
    j_ll,
    j_true,
    ll_derivative,
    ll_log_tail,
    ll_middle,
    ll_quadratic,
    ll_second_derivative,
    ll_value,
    log_tail_alpha,
    mu_max,
    transition_params,
)

__all__ = [
    "IRLSResult",
    "accept_step",
    "admissible_mu",
    "conservation_residual",
    "continuation_update",
    "dual_ssn",
    "flow_gap",
    "free_arc_mask",
    "initialize_smoothing",
    "irls",
    "irls_weights",
    "j_ll",
    "j_true",
    "ll_derivative",
    "ll_log_tail",
    "ll_middle",
    "ll_quadratic",
    "ll_second_derivative",
    "ll_value",
    "log_tail_alpha",
    "mu_max",
    "newton_diag",
    "newton_matrix",
    "node_arc_incidence",
    "predictor_y",
    "primal_from_dual",
    "refit_support",
    "transition_params",
]
