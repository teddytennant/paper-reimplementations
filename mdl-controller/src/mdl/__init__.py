"""Memory Decision Layer (MDL) geometric controller, arXiv:2609.22043."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mdl.actions import action_from_cfinal
from mdl.constants import (
    BIAS,
    D,
    EPS_C,
    GAINS,
    NMAX,
    NMIN,
    QR_SEED,
    SIGMA,
    SUBSPACE_DIMS,
    THETA,
)
from mdl.controller import Decision, decide, decide_from_memories
from mdl.encoding import value_encode
from mdl.fusion import confidence_consistency, fuse, gated_confidence, risk_gate
from mdl.signals import conflict_rate, extract_signals, pair_conflict
from mdl.subspaces import Subspaces, build_subspaces, subspace_weights

__all__ = [
    "BIAS",
    "D",
    "Decision",
    "EPS_C",
    "GAINS",
    "NMAX",
    "NMIN",
    "QR_SEED",
    "SIGMA",
    "SUBSPACE_DIMS",
    "Subspaces",
    "THETA",
    "action_from_cfinal",
    "build_subspaces",
    "confidence_consistency",
    "conflict_rate",
    "decide",
    "decide_from_memories",
    "extract_signals",
    "fuse",
    "gated_confidence",
    "pair_conflict",
    "risk_gate",
    "subspace_weights",
    "value_encode",
]

__version__ = "0.1.0"
