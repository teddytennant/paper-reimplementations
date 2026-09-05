"""Hyperparameters for CASTER.

The paper fixes tau = 0.8 globally (Sec. 4.3) and states that lambda, beta,
the confidence threshold and the coverage requirements are also fixed
globally, but it does not print their values, and the released PyTorch repo
does not ship `caster.methods.caster`. Everything except `tau` below is my
choice, picked so the synthetic demo behaves the way Sec. 4.4 and Sec. 4.6
describe: benign affine shifts pass, a 1000-classes-in-64-samples style batch
fails on coverage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CasterConfig:
    """Configuration for the fit and the test-time transport."""

    k_max: int = 64
    """Cap on the discriminative subspace, k0 = min(k_max, C - 1) in eq. (1)."""

    lam: float = 1e-3
    """Covariance shrinkage in eq. (3), added to both covariances before the roots."""

    beta: float = 1e-3
    """Class-covariance ridge in eq. (4)."""

    within_ridge: float = 1e-6
    """Relative ridge on the within-class scatter before the eq. (1) eigenproblem.

    Scaled by trace(S_w) / d, so it is a fraction of the average within-class
    variance rather than an absolute number that depends on feature units.
    """

    confidence: float = 0.9
    """Retention threshold in eq. (6), applied to softmax(g_c) from eq. (5)."""

    tau: float = 0.8
    """Certificate threshold in eq. (9). The paper's shipped value (Sec. 4.3)."""

    min_retained: int = 8
    """Minimum number of confident samples for the gate to consider accepting."""

    min_class_coverage: float = 0.25
    """Fraction of the C source classes that must be occupied in the retained set."""

    eps: float = 1e-12
    """The epsilon added to gamma_s in the denominator of eq. (9)."""
