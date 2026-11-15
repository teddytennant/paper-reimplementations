"""Stiefel-AdamW: AdamW on W=BA with A constrained to the Stiefel manifold."""

from stiefel_adamw.geometry import (
    cayley_omega,
    cayley_retract,
    orth_error,
    project,
    project_a1,
    qr_retract,
    random_stiefel,
    row_orth_error,
    skew,
    sym,
)
from stiefel_adamw.optimizer import (
    StiefelAdamWState,
    factor_gradients,
    factorization_loss,
    init,
    step,
    step_from_dW,
)

__all__ = [
    "StiefelAdamWState",
    "cayley_omega",
    "cayley_retract",
    "factor_gradients",
    "factorization_loss",
    "init",
    "orth_error",
    "project",
    "project_a1",
    "qr_retract",
    "random_stiefel",
    "row_orth_error",
    "skew",
    "step",
    "step_from_dW",
    "sym",
]
