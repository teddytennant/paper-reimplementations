"""TQSC: Transformer-based Quantum State Characterizer (arXiv:2609.20523)."""

from jax_tqsc.data import make_synthetic_dataset
from jax_tqsc.model import TQSCConfig, count_params, forward, init_tqsc
from jax_tqsc.physics import cholesky_rho, fidelity, frobenius_mse, rho_from_chol
from jax_tqsc.train import evaluate, train_tqsc

__version__ = "0.1.0"

__all__ = [
    "TQSCConfig",
    "cholesky_rho",
    "count_params",
    "evaluate",
    "fidelity",
    "forward",
    "frobenius_mse",
    "init_tqsc",
    "make_synthetic_dataset",
    "rho_from_chol",
    "train_tqsc",
]
