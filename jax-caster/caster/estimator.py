"""CASTER end to end: fit on source features, adapt one target batch at a time.

`Caster.fit` builds the state of Sec. 3 (subspace, class moments, priors,
pooled within-class covariance). `Caster.adapt` runs eq. (2)'s subspace
restriction, eq. (3)'s transport, eq. (5)'s discriminant, then eqs. (6) to (9)
and the gate, and returns the transported prediction only when the gate
accepts.

The paper's fallback is the frozen model head. This package has no backbone, so
pass `head_predictions` if you have a head; otherwise the fallback is the
untransported source Gaussian discriminant, which is the Source-GDA row of
Table 1.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .certificate import (
    Certificate,
    GateDecision,
    centroid_residuals,
    gate,
    residual_to_margin,
    retained_set,
)
from .config import CasterConfig
from .gda import confidence, discriminant_scores, predict, source_scores
from .source import SourceState, fit_source, min_class_separation
from .subspace import active_dim
from .transport import (
    AffineMap,
    estimate_transport,
    inverse_transport,
    transport_class_moments,
)


class BatchResult(NamedTuple):
    """What CASTER decided for one target batch."""

    predictions: jnp.ndarray            # what CASTER actually returns
    transported: jnp.ndarray            # argmax of eq. (5) on the transported moments
    fallback: jnp.ndarray               # frozen head, or Source-GDA if no head given
    decision: GateDecision
    affine: AffineMap
    k: int                              # eq. (2)
    z: jnp.ndarray                      # projected batch

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    @property
    def certificate(self) -> float:
        return float(self.decision.certificate.value)


class Caster:
    """The estimator. Holds the source state and the fixed thresholds."""

    def __init__(self, state: SourceState, config: CasterConfig = CasterConfig()):
        self.state = state
        self.config = config

    @classmethod
    def fit(cls, features, labels, num_classes, config: CasterConfig = CasterConfig()):
        return cls(fit_source(features, labels, num_classes, config), config)

    @property
    def num_classes(self) -> int:
        return self.state.num_classes

    def project(self, features, k=None):
        """z = U_k^T h, with U sliced to the first k directions as in eq. (2)."""
        k = self.state.k0 if k is None else k
        return jnp.asarray(features) @ self.state.basis[:, :k]

    def source_predictions(self, features):
        """The untransported Source-GDA baseline of Table 1."""
        state = self.state
        z = self.project(features)
        return predict(source_scores(state, z, self.config.beta))

    def gamma_s(self, k):
        """Equation (8) in the first k subspace coordinates."""
        state = self.state.restrict(k)
        present = np.asarray(state.counts) > 0
        return min_class_separation(state.means[present], state.pooled_cov)

    def adapt(self, features, head_predictions=None) -> BatchResult:
        """Adapt one target batch. See the module docstring for the fallback."""
        cfg = self.config
        h = jnp.asarray(features)
        batch_size = h.shape[0]
        k = active_dim(self.state.k0, batch_size)
        state = self.state.restrict(k)
        z = h @ state.basis

        affine = estimate_transport(state.global_mean, state.global_cov, z, cfg.lam)
        t_means, t_covs = transport_class_moments(affine, state.means, state.covs, cfg.beta)
        t_scores = discriminant_scores(z, t_means, t_covs, state.priors)
        transported = predict(t_scores)

        cert = self._certificate(state, z, affine, t_scores, k)
        decision = gate(
            cert, self.num_classes, cfg.tau, cfg.min_retained, cfg.min_class_coverage
        )

        if head_predictions is None:
            fallback = predict(source_scores(state, z, cfg.beta))
        else:
            fallback = jnp.asarray(head_predictions)

        predictions = transported if decision.accepted else fallback
        return BatchResult(
            predictions=predictions,
            transported=transported,
            fallback=fallback,
            decision=decision,
            affine=affine,
            k=k,
            z=z,
        )

    def _certificate(self, state, z, affine, t_scores, k) -> Certificate:
        cfg = self.config
        gamma = float(self.gamma_s(k))
        mask, pseudo = retained_set(
            confidence(t_scores), predict(t_scores), cfg.confidence
        )
        n_retained = int(mask.sum())
        if n_retained == 0:
            return Certificate(float("inf"), 0, 0, gamma, float("inf"))

        zbar = inverse_transport(affine, z[jnp.asarray(mask)])
        residuals, occupied = centroid_residuals(zbar, pseudo, state.means, self.num_classes)
        value, numerator = residual_to_margin(
            jnp.asarray(residuals), occupied, state.pooled_cov, gamma, cfg.eps
        )
        return Certificate(
            value=float(value),
            n_retained=n_retained,
            n_occupied=int(occupied.sum()),
            gamma_s=gamma,
            residual=float(numerator),
        )
