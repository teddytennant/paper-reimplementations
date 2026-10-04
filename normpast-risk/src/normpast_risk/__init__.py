"""NormPaST-Risk: segment-level Alzheimer's risk on handwriting trajectories."""

from normpast_risk.loss import (
    LAMBDA_NORM,
    LAMBDA_SEG,
    classification_loss,
    normative_loss,
    segment_loss,
    total_loss,
)
from normpast_risk.model import (
    DEFAULTS,
    NormPaSTRisk,
    apply,
    init_params,
    paper_air_gate,
    paper_air_modulate,
    ssm_scan,
    ssm_step,
    transition_A,
)
from normpast_risk.segments import aggregate_segments, n_segments, project_tokens

__all__ = [
    "DEFAULTS",
    "LAMBDA_NORM",
    "LAMBDA_SEG",
    "NormPaSTRisk",
    "aggregate_segments",
    "apply",
    "classification_loss",
    "init_params",
    "n_segments",
    "normative_loss",
    "paper_air_gate",
    "paper_air_modulate",
    "project_tokens",
    "segment_loss",
    "ssm_scan",
    "ssm_step",
    "total_loss",
    "transition_A",
]

__version__ = "0.1.0"
