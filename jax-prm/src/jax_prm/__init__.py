"""Persistent Recurrent Memory (PRM) between transformer layers, arXiv:2609.17251."""

from jax_prm.config import PRMConfig
from jax_prm.gru import gru, init_gru
from jax_prm.prm import gated_add, init_memory, init_prm, prm_forward
from jax_prm.train import adamw_init, adamw_step, loss_fn
from jax_prm.transformer import count_params, forward, init

__version__ = "0.1.0"

__all__ = [
    "PRMConfig",
    "adamw_init",
    "adamw_step",
    "count_params",
    "forward",
    "gated_add",
    "gru",
    "init",
    "init_gru",
    "init_memory",
    "init_prm",
    "loss_fn",
    "prm_forward",
]
