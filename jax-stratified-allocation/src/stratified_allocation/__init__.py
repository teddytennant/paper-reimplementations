"""Design-based sampling theory of arXiv:2609.04420, Section 6.

The paper's identities (Propositions 5, 6, 7, Theorem 8, Corollaries 10 and 12) are
exact statements about a finite population, and several of them are differences of two
nearly equal sums. They are checked here in double precision, so importing this package
turns on jax's x64 mode. Do that before any array is created, which is why it happens at
import time rather than being left to the caller.
"""

import jax

jax.config.update("jax_enable_x64", True)

from stratified_allocation.population import (  # noqa: E402
    Population,
    class_counts,
    class_means,
    class_dispersions,
    between_class_term,
    pooled_dispersion,
    weighted_losses,
    equidispersed_population,
    random_population,
)
from stratified_allocation.designs import (  # noqa: E402
    balanced_allocation,
    neyman_allocation,
    proportional_allocation,
    stratified_estimate,
    stratified_sample,
    uniform_estimate,
    uniform_sample,
)
from stratified_allocation.variance import (  # noqa: E402
    allocation_efficiency_loss,
    efficiency_prediction,
    equidispersed_relative_efficiency,
    realised_allocation_ratio,
    uniform_minus_proportional,
    variance_stratified,
    variance_stratified_population,
    variance_uniform,
    variance_uniform_population,
)
from stratified_allocation.bounds import serfling_uniform_bound  # noqa: E402
from stratified_allocation.representatives import (  # noqa: E402
    cluster_representatives,
    kmeans_labels,
    representative_design,
    representative_estimate,
    representative_bias,
)
from stratified_allocation.rankmap import (  # noqa: E402
    average_precision,
    conformal_rank_map,
    rank_map,
    roc_auc,
)

__all__ = [
    "Population",
    "allocation_efficiency_loss",
    "average_precision",
    "balanced_allocation",
    "between_class_term",
    "class_counts",
    "class_dispersions",
    "class_means",
    "cluster_representatives",
    "conformal_rank_map",
    "efficiency_prediction",
    "equidispersed_population",
    "equidispersed_relative_efficiency",
    "kmeans_labels",
    "neyman_allocation",
    "pooled_dispersion",
    "proportional_allocation",
    "random_population",
    "rank_map",
    "realised_allocation_ratio",
    "representative_bias",
    "representative_design",
    "representative_estimate",
    "roc_auc",
    "serfling_uniform_bound",
    "stratified_estimate",
    "stratified_sample",
    "uniform_estimate",
    "uniform_minus_proportional",
    "uniform_sample",
    "variance_stratified",
    "variance_stratified_population",
    "variance_uniform",
    "variance_uniform_population",
]
