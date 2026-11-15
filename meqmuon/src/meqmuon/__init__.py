"""Matrix-equilibrating Muon."""

from meqmuon.normalize import (
    coefficient_of_variation,
    column_normalize,
    column_rms,
    global_rms_normalize,
    one_sided_normalize,
    row_normalize,
    row_rms,
)
from meqmuon.orthogonalize import newton_schulz
from meqmuon.step import init_momentum, meqmuon_step, step_pytree

__all__ = [
    "newton_schulz",
    "row_rms",
    "column_rms",
    "coefficient_of_variation",
    "row_normalize",
    "column_normalize",
    "global_rms_normalize",
    "one_sided_normalize",
    "meqmuon_step",
    "init_momentum",
    "step_pytree",
]
