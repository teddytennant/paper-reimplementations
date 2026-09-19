"""jax-pdeg: single-loop projected damped extragradient (arXiv:2609.17973)."""

from jax_pdeg.params import (
    PDEGparams,
    init_energy_bound,
    params_from_L_mu_tau,
    pdeg_params,
    warmup_length,
)
from jax_pdeg.pdeg import (
    PDEGresult,
    PDEGstate,
    G,
    certificate_S,
    grad_G,
    init_state,
    run_pdeg,
    run_pdeg_warmup,
    step,
)
from jax_pdeg.problems import (
    MinimaxProblem,
    bilinear_saddle,
    make_problem,
    ncc_interval,
    quadratic_scsc,
)
from jax_pdeg.projections import (
    project_ball,
    project_box,
    project_simplex,
    project_unconstrained,
)

__version__ = "0.1.0"

__all__ = [
    "G",
    "MinimaxProblem",
    "PDEGparams",
    "PDEGresult",
    "PDEGstate",
    "bilinear_saddle",
    "certificate_S",
    "grad_G",
    "init_energy_bound",
    "init_state",
    "make_problem",
    "ncc_interval",
    "params_from_L_mu_tau",
    "pdeg_params",
    "project_ball",
    "project_box",
    "project_simplex",
    "project_unconstrained",
    "quadratic_scsc",
    "run_pdeg",
    "run_pdeg_warmup",
    "step",
    "warmup_length",
]
