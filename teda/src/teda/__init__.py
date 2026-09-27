"""TEDA: transmittance extraction and distance alignment (arXiv:2609.28967)."""

from teda.extract import extract_transmittance, gate, noise_scale_tau_g
from teda.isrf import isrf_apply
from teda.match import contrast_weights, match_distance, run_batch, run_pixel
from teda.operators import (
    ETA_HIGH,
    ETA_LOW,
    Stage1Ops,
    build_stage1_operators,
    second_difference_operator,
)
from teda.planck import linear_in_T, planck_B
from teda.synth import observe_radiance, synthetic_kappa, transmittance_bank

__all__ = [
    "ETA_HIGH",
    "ETA_LOW",
    "Stage1Ops",
    "build_stage1_operators",
    "contrast_weights",
    "extract_transmittance",
    "gate",
    "isrf_apply",
    "linear_in_T",
    "match_distance",
    "noise_scale_tau_g",
    "observe_radiance",
    "planck_B",
    "run_batch",
    "run_pixel",
    "second_difference_operator",
    "synthetic_kappa",
    "transmittance_bank",
]
