"""CASTER: gradient-free affine statistics transport with a transportability certificate.

JAX implementation of arXiv:2609.00374, Sec. 3. Equation numbers in the
docstrings refer to that paper.

    caster.subspace     eq. (1) discriminative subspace, eq. (2) active dimension
    caster.source       the retained source state, eq. (8) gamma_s
    caster.transport    eq. (3) whiten-color map, eq. (4) transported moments, eq. (6)
    caster.gda          eq. (5) Gaussian discriminant
    caster.certificate  eqs. (7) and (9), and the accept rule of Sec. 3.2
    caster.estimator    the two of them wired together
"""

from .certificate import (
    Certificate,
    GateDecision,
    centroid_residuals,
    gate,
    mixed_metric_certificate,
    residual_to_margin,
)
from .config import CasterConfig
from .estimator import BatchResult, Caster
from .gda import confidence, discriminant_scores, predict
from .source import SourceState, fit_source, min_class_separation
from .subspace import active_dim, lda_directions, scatter_matrices, subspace_dim
from .transport import (
    AffineMap,
    empirical_moments,
    estimate_transport,
    inverse_transport,
    transport_class_moments,
    use_diagonal_form,
    whiten_color_map,
)

__all__ = [
    "AffineMap",
    "BatchResult",
    "Caster",
    "CasterConfig",
    "Certificate",
    "GateDecision",
    "SourceState",
    "active_dim",
    "centroid_residuals",
    "confidence",
    "discriminant_scores",
    "empirical_moments",
    "estimate_transport",
    "fit_source",
    "gate",
    "inverse_transport",
    "lda_directions",
    "min_class_separation",
    "mixed_metric_certificate",
    "predict",
    "residual_to_margin",
    "scatter_matrices",
    "subspace_dim",
    "transport_class_moments",
    "use_diagonal_form",
    "whiten_color_map",
]
