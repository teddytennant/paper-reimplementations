"""Online Self-Weighted Fine-Tuning (arXiv:2609.00734) in JAX.

For each training query, draw K inference-only rollouts, let p_hat(q) be the
fraction the verifier accepts, and scale that query's SFT loss by w = 1 - p_hat.
The gradient direction stays on the expert trajectory o*; only the magnitude
moves. Start at `weighting.py` for the estimator and `train.py` for Algorithm 1.

Algorithm 1 lives in `osw_ft.train`, imported separately so that
`python -m osw_ft.train` does not re-enter the package.
"""

from .loss import g0, g0_sq_norms, sft_loss, weighted_sft_loss
from .model import ModelConfig, forward, init_params, log_probs
from .rollout import estimate_p_hat, estimate_weights, rollout_rewards, success_probability
from .task import build_dataset, exact_match_reward
from .weighting import (
    expected_squared_weight,
    expected_weight,
    osw_weight,
    p_hat,
    sigma_squared,
    sigma_squared_sft,
    signal_loss_probability,
)

__all__ = [
    "ModelConfig",
    "build_dataset",
    "estimate_p_hat",
    "estimate_weights",
    "exact_match_reward",
    "expected_squared_weight",
    "expected_weight",
    "forward",
    "g0",
    "g0_sq_norms",
    "init_params",
    "log_probs",
    "osw_weight",
    "p_hat",
    "rollout_rewards",
    "sft_loss",
    "sigma_squared",
    "sigma_squared_sft",
    "signal_loss_probability",
    "success_probability",
    "weighted_sft_loss",
]
