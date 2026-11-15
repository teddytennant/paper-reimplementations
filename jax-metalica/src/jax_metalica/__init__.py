"""METALICA replica-exchange metadynamics sampler (arXiv:2609.17823)."""

from jax_metalica.bias import CVGrid, cv_grid, deposit_hills, evaluate_V
from jax_metalica.gmm import ToyGMM, log_p_t, responsibilities, score, toy_gmm, xhat0
from jax_metalica.kernels import (
    forward_mean,
    forward_var,
    grad_reward,
    incremental_log_weight,
    log_B0_minus_log_B,
    log_normal,
    log_pi,
    pretrained_backward_mean,
    reward,
    tilted_backward_mean,
)
from jax_metalica.mbar import (
    free_energy_difference,
    mbar_weights,
    solve_mbar,
    true_delta_F,
)
from jax_metalica.metalica import MetalicaConfig, MetalicaResult, run
from jax_metalica.replica import attempt_swap, init_chain, replica_step, swap_acceptance
from jax_metalica.sde import a_t, beta, guidance_lambda, nu_t, time_grid

__all__ = [
    "CVGrid",
    "MetalicaConfig",
    "MetalicaResult",
    "ToyGMM",
    "a_t",
    "attempt_swap",
    "beta",
    "cv_grid",
    "deposit_hills",
    "evaluate_V",
    "forward_mean",
    "forward_var",
    "free_energy_difference",
    "grad_reward",
    "guidance_lambda",
    "incremental_log_weight",
    "init_chain",
    "log_B0_minus_log_B",
    "log_normal",
    "log_p_t",
    "log_pi",
    "mbar_weights",
    "nu_t",
    "pretrained_backward_mean",
    "replica_step",
    "responsibilities",
    "reward",
    "run",
    "score",
    "solve_mbar",
    "swap_acceptance",
    "tilted_backward_mean",
    "time_grid",
    "toy_gmm",
    "true_delta_F",
    "xhat0",
]
