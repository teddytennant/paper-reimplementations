import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

from lazy_so.core import (
    CountingOracle,
    BISECTION_CAP,
    cubic_gradient,
    cubic_hessian,
    cubic_proximal,
    cubic_regularized_newton,
    crn_step,
    extrapolation_params,
    finite_seg_search,
    lazy_iteration_cost,
    pagd_iteration_count,
    pagd_seg,
    proximal_epoch_count,
    search_free_ms_acceleration,
)

__all__ = [
    "BISECTION_CAP",
    "CountingOracle",
    "crn_step",
    "cubic_gradient",
    "cubic_hessian",
    "cubic_proximal",
    "cubic_regularized_newton",
    "extrapolation_params",
    "finite_seg_search",
    "lazy_iteration_cost",
    "pagd_iteration_count",
    "pagd_seg",
    "proximal_epoch_count",
    "search_free_ms_acceleration",
]
