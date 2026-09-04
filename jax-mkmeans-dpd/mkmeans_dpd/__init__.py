"""JAX implementation of arXiv:2608.30093v1, Mondal, Banerjee and Mandal,
"Robust K-means Clustering using the Density Power Divergence Measure".

Equation and algorithm numbers in the docstrings throughout this package refer
to that paper.
"""

from .baselines import KMeansResult, MKMeansResult, kmeans, kmeans_plusplus, mkmeans
from .cluster import DPDResult, dc_mkmeans_dpd, mkmeans_dpd
from .data import cluster_accuracy, load_iris, make_simulated
from .dpd import (
    ClusterSpec,
    dpd_gradients,
    dpd_objective,
    joint_objective,
    mahalanobis,
    mdpde,
    pointwise_loss,
    spec_from_cov,
    weighted_step,
)
from .indices import (
    calinski_harabasz,
    davies_bouldin,
    median_davies_bouldin,
    trimmed_calinski_harabasz,
)
from .init import core_init

__all__ = [
    "ClusterSpec",
    "DPDResult",
    "KMeansResult",
    "MKMeansResult",
    "calinski_harabasz",
    "cluster_accuracy",
    "core_init",
    "davies_bouldin",
    "dc_mkmeans_dpd",
    "dpd_gradients",
    "dpd_objective",
    "joint_objective",
    "kmeans",
    "kmeans_plusplus",
    "load_iris",
    "mahalanobis",
    "make_simulated",
    "mdpde",
    "median_davies_bouldin",
    "mkmeans",
    "mkmeans_dpd",
    "pointwise_loss",
    "spec_from_cov",
    "trimmed_calinski_harabasz",
    "weighted_step",
]
