"""Amortized Gromov-Wasserstein matching for supervised graph prediction."""

from gwsgp.data import (
    canonical_graphs,
    complete_adjacency,
    complete_bipartite_adjacency,
    cycle_adjacency,
    path_adjacency,
    permute_adjacency,
    sample_batch,
)
from gwsgp.gw import greedy_permutation, gw_loss, nested_gw, permutation_matrix
from gwsgp.matcher import match, omega_bistochastic
from gwsgp.predictor import predict_adjacency
from gwsgp.sinkhorn import sinkhorn
from gwsgp.train import init_params, sgp_loss_one, train

__all__ = [
    "canonical_graphs",
    "complete_adjacency",
    "complete_bipartite_adjacency",
    "cycle_adjacency",
    "greedy_permutation",
    "gw_loss",
    "init_params",
    "match",
    "nested_gw",
    "omega_bistochastic",
    "path_adjacency",
    "permutation_matrix",
    "permute_adjacency",
    "predict_adjacency",
    "sample_batch",
    "sgp_loss_one",
    "sinkhorn",
    "train",
]
