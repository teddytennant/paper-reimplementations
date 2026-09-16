"""JAX kernel of Branched Optimal Transport Amortization (arXiv:2609.15072)."""

from jax_bota.cost import (
    atomic_branched_cost,
    edge_transport_cost,
    soft_atomic_cost,
    two_source_two_sink_costs,
)
from jax_bota.flow import bfm_loss, bfm_loss_from_vectors, mlp_apply, mlp_init, path_interpolant
from jax_bota.particles import euler_trajectories, normalize_masses, solve_discrete
from jax_bota.train import sample_gmm_pair, train_bfm, train_bota

__version__ = "0.1.0"
__all__ = [
    "atomic_branched_cost",
    "bfm_loss",
    "bfm_loss_from_vectors",
    "edge_transport_cost",
    "euler_trajectories",
    "mlp_apply",
    "mlp_init",
    "normalize_masses",
    "path_interpolant",
    "sample_gmm_pair",
    "soft_atomic_cost",
    "solve_discrete",
    "train_bfm",
    "train_bota",
    "two_source_two_sink_costs",
]
