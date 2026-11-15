"""Musec: MomentUm SpEctral Clipping (arXiv:2609.11655)."""

from musec.clip import (
    hard_spectral_clip,
    soft_saturation,
    soft_spectral_clip,
    soft_spectral_clip_exact,
    spectral_clip,
)
from musec.optimizer import MusecState, init, step

__all__ = [
    "MusecState",
    "hard_spectral_clip",
    "init",
    "soft_saturation",
    "soft_spectral_clip",
    "soft_spectral_clip_exact",
    "spectral_clip",
    "step",
]
