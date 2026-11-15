"""JAX kernel of disintegrated PAC-Bayes self-bounding majority votes."""

from jax_dis_smv.bounds import (
    binary_kl,
    dis_r_certificate,
    dis_r_penalty,
    dis_v_certificate,
    dis_v_penalty,
    kl_inv,
)
from jax_dis_smv.dirichlet import (
    dirichlet_kl,
    dirichlet_renyi,
    log_beta,
    log_density_ratio,
    log_dirichlet_pdf,
    sample_dirichlet,
)
from jax_dis_smv.majority_vote import (
    empirical_risk_01,
    empirical_risk_sigmoid,
    majority_vote_predict,
    vote_scores,
)
from jax_dis_smv.train import (
    TrainResult,
    algorithm_1,
    axis_aligned_stumps,
    split_data,
    synthetic_binary_2d,
)

__all__ = [
    "TrainResult",
    "algorithm_1",
    "axis_aligned_stumps",
    "binary_kl",
    "dirichlet_kl",
    "dirichlet_renyi",
    "dis_r_certificate",
    "dis_r_penalty",
    "dis_v_certificate",
    "dis_v_penalty",
    "empirical_risk_01",
    "empirical_risk_sigmoid",
    "kl_inv",
    "log_beta",
    "log_density_ratio",
    "log_dirichlet_pdf",
    "majority_vote_predict",
    "sample_dirichlet",
    "split_data",
    "synthetic_binary_2d",
    "vote_scores",
]
