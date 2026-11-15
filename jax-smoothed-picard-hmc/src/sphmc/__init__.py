"""Smoothed Picard Hamiltonian Monte Carlo (arXiv:2609.06906)."""

from sphmc.kernel import (
    PicardState,
    make_exact_smoothed_quadratic_score,
    make_quadratic_score,
    make_stochastic_score,
    picard_hmc_step,
    stochastic_score,
)
from sphmc.prox import prox_gradient_descent, prox_quadratic
from sphmc.quadrature import ChebyshevLobatto, chebyshev_lobatto_nodes, quadrature_weights
from sphmc.sampler import SPHMCConfig, rgo_quadratic, sample_quadratic, sphmc_trajectory

__all__ = [
    "ChebyshevLobatto",
    "PicardState",
    "SPHMCConfig",
    "chebyshev_lobatto_nodes",
    "make_exact_smoothed_quadratic_score",
    "make_quadratic_score",
    "make_stochastic_score",
    "picard_hmc_step",
    "prox_gradient_descent",
    "prox_quadratic",
    "quadrature_weights",
    "rgo_quadratic",
    "sample_quadratic",
    "sphmc_trajectory",
    "stochastic_score",
]
