"""CoverPruner: visual token pruning as Representational Coverage Maximization.

Implements the selector from "Who Speaks for the Pruned? Visual Token Pruning as
Coverage Optimization" (Zhu et al., arXiv:2609.03158), EMNLP 2026.

The paper's pipeline is Algorithm 1: project the visual tokens, build calibrated
coverage utilities A (Eq. 3), read query demand w from a layer-0 attention probe
(Eq. 4), then pick the retained set by marginal coverage gain (Eq. 6). Everything
here is the selection half of that. Features and the probe come in as arrays.

x64 is switched on at import. The greedy step is an argmax over marginal gains
that can be close together, and cosine similarities feeding it come out of a
matmul; in float32 that matmul runs at TF32 on an H200 and the reduced mantissa
is enough to swap two candidates. float64 keeps CPU and GPU selections identical.
"""

import jax

jax.config.update("jax_enable_x64", True)

from coverpruner.coverage import (  # noqa: E402
    calibrated_utilities,
    centered_similarity,
    coverage,
    similarity_matrix,
    total_coverage,
)
from coverpruner.relevance import (  # noqa: E402
    demand_weights,
    layer0_attention_probe,
    relevance_from_probe,
)
from coverpruner.select import (  # noqa: E402
    Selection,
    greedy_select,
    lazy_greedy_select,
    marginal_gains,
    select_tokens,
)

__version__ = "0.1.0"

__all__ = [
    "Selection",
    "calibrated_utilities",
    "centered_similarity",
    "coverage",
    "demand_weights",
    "greedy_select",
    "layer0_attention_probe",
    "lazy_greedy_select",
    "marginal_gains",
    "relevance_from_probe",
    "select_tokens",
    "similarity_matrix",
    "total_coverage",
]
