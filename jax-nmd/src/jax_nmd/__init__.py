"""JAX kernel of Neural Modal Decomposition (arXiv:2609.14402)."""

from jax_nmd.config import NMDConfig
from jax_nmd.couplings import amp_net_ports, complex_from_mag_phase
from jax_nmd.loss import ell_gauge, gauge_loss, log_z_loss, scattering_mae, total_loss
from jax_nmd.poles import poles_from_raw
from jax_nmd.synthesize import dominance_score, synthesize_z, z_to_s
from jax_nmd.train import forward, init_params, make_synthetic_batch, train_step

__all__ = [
    "NMDConfig",
    "amp_net_ports",
    "complex_from_mag_phase",
    "dominance_score",
    "ell_gauge",
    "forward",
    "gauge_loss",
    "init_params",
    "log_z_loss",
    "make_synthetic_batch",
    "poles_from_raw",
    "scattering_mae",
    "synthesize_z",
    "total_loss",
    "train_step",
    "z_to_s",
]
