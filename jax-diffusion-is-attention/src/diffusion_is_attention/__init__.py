"""JAX implementation of arXiv:2609.00885, the diffusion / attention equivalence.

Module map, by section of the paper:

  closed_form   Sec. II-C, eqs (7)-(15). The optimal one-step denoiser, its
                dot-product attention form, and softmax as a special case.
  score         Sec. II-B, eqs (2)-(5). The Gaussian mixture, its score, and the
                probability-flow ODE.
  convergence   Sec. II-D. The decreasing-sigma sweep and Figure 2.
  tree          Sec. IV. Tree cross attention.
  generate      Sec. V. Algorithms 1 and 2.
"""

from .closed_form import (
    attention_weights,
    dot_product_attention,
    dot_product_attention_weights,
    gaussian_likelihood,
    one_hot_dataset,
    optimal_denoiser,
    qp_objective,
    softmax_as_attention,
    squared_distances,
)
from .convergence import (
    Trajectory,
    figure2_setup,
    reverse_attention_trajectory,
    sigma_schedule,
    weight_entropy,
)
from .generate import IDENTITY_CODEC, Codec, Generation, generate, spatial_attention
from .score import (
    mixture_log_density,
    mixture_weights,
    posterior_mean,
    posterior_mean_via_closed_form,
    probability_flow_rhs,
    reverse_ode,
    score,
    sigma_from_alpha,
)
from .tree import SimilarityTree, TopK, exact_topk, tree_attention

__version__ = "0.1.0"

__all__ = [
    "Codec",
    "Generation",
    "IDENTITY_CODEC",
    "SimilarityTree",
    "TopK",
    "Trajectory",
    "attention_weights",
    "dot_product_attention",
    "dot_product_attention_weights",
    "exact_topk",
    "figure2_setup",
    "gaussian_likelihood",
    "generate",
    "mixture_log_density",
    "mixture_weights",
    "one_hot_dataset",
    "optimal_denoiser",
    "posterior_mean",
    "posterior_mean_via_closed_form",
    "probability_flow_rhs",
    "qp_objective",
    "reverse_attention_trajectory",
    "reverse_ode",
    "score",
    "sigma_from_alpha",
    "sigma_schedule",
    "softmax_as_attention",
    "spatial_attention",
    "squared_distances",
    "tree_attention",
    "weight_entropy",
]
