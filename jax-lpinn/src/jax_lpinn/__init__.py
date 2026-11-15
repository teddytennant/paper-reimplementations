"""lPINN: linearized physics-informed neural network (arXiv:2609.14926)."""

from jax_lpinn.model import (
    LPINNConfig,
    fluct_value,
    init_fluct_hidden,
    init_mean_params,
    last_hidden,
    mean_value,
    n_eta,
    predict,
)
from jax_lpinn.physics import ade_operator, ogata_banks
from jax_lpinn.metrics import rrmse
from jax_lpinn.train import (
    build_ade_lstsq_system,
    fit_reconstruction_W,
    inverse_ade,
    online_forward_lstsq,
    online_objective,
    train_fluct_derivative_matching,
    train_mean_derivative_matching,
    train_residual_based,
)

__all__ = [
    "LPINNConfig",
    "ade_operator",
    "build_ade_lstsq_system",
    "fit_reconstruction_W",
    "fluct_value",
    "init_fluct_hidden",
    "init_mean_params",
    "inverse_ade",
    "last_hidden",
    "mean_value",
    "n_eta",
    "ogata_banks",
    "online_forward_lstsq",
    "online_objective",
    "predict",
    "rrmse",
    "train_fluct_derivative_matching",
    "train_mean_derivative_matching",
    "train_residual_based",
]
