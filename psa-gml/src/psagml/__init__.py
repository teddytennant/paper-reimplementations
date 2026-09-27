"""PSA-GML JAX kernel (arXiv:2609.29150)."""

from psagml.channels import complex_normal, rayleigh_channels
from psagml.lstm import LSTMParams, init_lstm, lstm_step, preprocess_grad
from psagml.pso import random_phase_zf_wsr, run_pso
from psagml.star_ris import (
    amplitude_split,
    composite_channel,
    phi_to_tilde,
    star_coefficients,
    unconstrained_phi,
)
from psagml.train_meta import (
    meta_loss,
    meta_train,
    pack_xi,
    pso_to_xi,
    refine,
    run_psagml,
    unpack_xi,
    wsr_from_xi,
)
from psagml.wsr import sinr, weighted_sum_rate
from psagml.zf import collapsed_precoder, power_normalize, zf_collapsed_X, zf_precoder

__all__ = [
    "LSTMParams",
    "amplitude_split",
    "collapsed_precoder",
    "complex_normal",
    "composite_channel",
    "init_lstm",
    "lstm_step",
    "meta_loss",
    "meta_train",
    "pack_xi",
    "phi_to_tilde",
    "power_normalize",
    "preprocess_grad",
    "pso_to_xi",
    "random_phase_zf_wsr",
    "rayleigh_channels",
    "refine",
    "run_psagml",
    "run_pso",
    "sinr",
    "star_coefficients",
    "unconstrained_phi",
    "unpack_xi",
    "weighted_sum_rate",
    "wsr_from_xi",
    "zf_collapsed_X",
    "zf_precoder",
]
