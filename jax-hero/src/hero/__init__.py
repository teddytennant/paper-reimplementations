"""HeRo: history-aware FFN routing with kernelized linear-attention memory."""

from hero.config import HeroConfig
from hero.ops import phi, routed_residual, skip_loss
from hero.model import init_params, forward

__all__ = [
    "HeroConfig",
    "phi",
    "routed_residual",
    "skip_loss",
    "init_params",
    "forward",
]
