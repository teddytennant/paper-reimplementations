"""AdamX: cosine-controlled AMSGrad (arXiv:2609.11867)."""

from adamx.opt import (
    amsgrad_envelope,
    apply_updates,
    cosine_controller,
    init,
    update,
)

__all__ = [
    "amsgrad_envelope",
    "apply_updates",
    "cosine_controller",
    "init",
    "update",
]
