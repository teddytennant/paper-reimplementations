"""The transportability certificate, eqs. (6) to (9), and the gate around it.

Retain the target samples whose transported-GDA confidence clears a fixed
threshold, inverse-transport each one to source coordinates,

    zbar_i = A_t^-1 (z_i - b_t),                                        (6)

take the centroid residual of every occupied pseudo-class,

    delta_c = mean_{i in I_c} zbar_i - mu_c,                            (7)

and divide the largest of those residuals by the smallest source-class gap,

    gamma_s = min_{c != c'} ((mu_c - mu_c')^T Sigma_s^-1 (mu_c - mu_c'))^{1/2},   (8)
    r_t     = max_{c: |I_c| > 0} (delta_c^T Sigma_s^-1 delta_c)^{1/2} / (gamma_s + eps).  (9)

Both the numerator and the denominator are Mahalanobis lengths in the same
pooled within-class covariance Sigma_s. That is the whole point of the design:
send z -> Lz for any invertible L, so mu -> L mu, Sigma_s -> L Sigma_s L^T and
delta -> L delta, and both lengths pick up the same factor, so r_t does not
move. `mixed_metric_certificate` below is the version the paper says fails, a
Euclidean numerator over a Mahalanobis denominator, kept so the difference can
be tested rather than asserted. See tests/test_certificate.py.

One caveat the paper does not spell out. That argument needs delta to transform
as L delta, which needs A_t to transform as L A_t L^-1. The symmetric PSD root
in eq. (3) is not equivariant under a general congruence, so re-basing the
subspace coordinates by hand does not leave r_t alone. What does is rescaling
the raw features, since the eq. (1) projection is contravariant and z never
moves at all; that is the case the paper is arguing about, and it is what
tests/test_certificate.py checks end to end.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .linalg import mahalanobis_sq


class Certificate(NamedTuple):
    """The certificate value and the evidence the gate weighs alongside it."""

    value: float          # r_t of eq. (9)
    n_retained: int       # |union_c I_c|
    n_occupied: int       # number of pseudo-classes with |I_c| > 0
    gamma_s: float        # eq. (8)
    residual: float       # the numerator of eq. (9), before dividing


def retained_set(confidences, predictions, threshold):
    """The retained samples of eq. (6) and their pseudo-class assignment."""
    mask = np.asarray(confidences) > threshold
    return mask, np.asarray(predictions)[mask]


def centroid_residuals(zbar, pseudo_labels, means, num_classes):
    """Equation (7) for every occupied pseudo-class.

    Returns (residuals, occupied) where `residuals` is (C, k) with zero rows
    for the unoccupied classes and `occupied` is a boolean (C,).
    """
    zbar = np.asarray(zbar)
    labels = np.asarray(pseudo_labels)
    means = np.asarray(means)
    residuals = np.zeros_like(means)
    occupied = np.zeros(num_classes, dtype=bool)
    for c in range(num_classes):
        rows = labels == c
        if not rows.any():
            continue
        occupied[c] = True
        residuals[c] = zbar[rows].mean(axis=0) - means[c]
    return residuals, occupied


def residual_to_margin(residuals, occupied, pooled_cov, gamma_s, eps):
    """Equation (9) from the pieces, so it can be tested on its own."""
    lengths = jnp.sqrt(jnp.maximum(mahalanobis_sq(pooled_cov, residuals), 0.0))
    lengths = jnp.where(jnp.asarray(occupied), lengths, -jnp.inf)
    numerator = jnp.max(lengths)
    return numerator / (gamma_s + eps), numerator


def mixed_metric_certificate(residuals, occupied, pooled_cov, gamma_s, eps):
    """The rejected variant: Euclidean numerator, Mahalanobis denominator.

    Sec. 3.2 says this ratio changes under an invertible rescaling of the
    feature coordinates and therefore stops measuring transport quality. It is
    here only as a control.
    """
    lengths = jnp.linalg.norm(jnp.asarray(residuals), axis=-1)
    lengths = jnp.where(jnp.asarray(occupied), lengths, -jnp.inf)
    return jnp.max(lengths) / (gamma_s + eps)


class GateDecision(NamedTuple):
    """Why the gate accepted or rejected a batch."""

    accepted: bool
    certificate: Certificate
    reason: str


def gate(certificate: Certificate, num_classes, tau, min_retained, min_class_coverage):
    """Sec. 3.2's accept rule: r_t <= tau plus the two coverage requirements."""
    required_classes = max(1, math.ceil(min_class_coverage * num_classes))
    if certificate.n_occupied == 0:
        return GateDecision(False, certificate, "no confident pseudo-class")
    if certificate.n_retained < min_retained:
        return GateDecision(
            False, certificate,
            f"retained {certificate.n_retained} < {min_retained} samples",
        )
    if certificate.n_occupied < required_classes:
        return GateDecision(
            False, certificate,
            f"covered {certificate.n_occupied} < {required_classes} classes",
        )
    if not (certificate.value <= tau):
        return GateDecision(False, certificate, f"r_t = {certificate.value:.3f} > tau = {tau}")
    return GateDecision(True, certificate, f"r_t = {certificate.value:.3f} <= tau = {tau}")
