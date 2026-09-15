"""GVBLL: GraphSAGE encoder plus variational Bayesian last layer (arXiv:2609.13655)."""

from gvbll.data import two_community_graph
from gvbll.elbo import elbo, expected_log_likelihood, map_nll, neg_elbo
from gvbll.encoder import encode, init_graphsage
from gvbll.online import online_laplace_update, posterior_from_vbll
from gvbll.train import init_gvbll, train, train_step
from gvbll.vbll import (
    init_vbll,
    kl_divergence,
    predict_map,
    predict_mc,
    sample_weights,
    std_from_log_std,
    uncertainty,
)

__all__ = [
    "encode",
    "elbo",
    "expected_log_likelihood",
    "init_graphsage",
    "init_gvbll",
    "init_vbll",
    "kl_divergence",
    "map_nll",
    "neg_elbo",
    "online_laplace_update",
    "posterior_from_vbll",
    "predict_map",
    "predict_mc",
    "sample_weights",
    "std_from_log_std",
    "train",
    "train_step",
    "two_community_graph",
    "uncertainty",
]
