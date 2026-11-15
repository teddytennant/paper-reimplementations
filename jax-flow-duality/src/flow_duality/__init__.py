"""Flow duality and source geometry for categorical generation (arXiv:2609.10863)."""

from flow_duality.coefficients import (
    continuous_ratio,
    k_gaussian,
    k_negexp,
    k_t,
    k_uniform,
)
from flow_duality.dfm import dfm_cross_entropy, predict_logits, sample_xt, train_tiny
from flow_duality.duality import induced_coefficient_mc
from flow_duality.interpolant import (
    argmax_tokens,
    convex_interpolant,
    one_hot_tokens,
    sample_distinct_token,
)
from flow_duality.sources import (
    sample_bounded_uniform,
    sample_centered_negexp,
    sample_gaussian,
    sample_source,
)

__version__ = "0.1.0"

__all__ = [
    "argmax_tokens",
    "continuous_ratio",
    "convex_interpolant",
    "dfm_cross_entropy",
    "induced_coefficient_mc",
    "k_gaussian",
    "k_negexp",
    "k_t",
    "k_uniform",
    "one_hot_tokens",
    "predict_logits",
    "sample_bounded_uniform",
    "sample_centered_negexp",
    "sample_distinct_token",
    "sample_gaussian",
    "sample_source",
    "sample_xt",
    "train_tiny",
]
