"""One-atom Iterative Atom Refinement (arXiv:2609.23812)."""

import jax

jax.config.update("jax_enable_x64", True)

from iar.data import combinatorial_codes, orthogonal_dictionary
from iar.refine import (
    multi_restart_recovery,
    one_atom_iar,
    refine_step,
    selection_counts,
    selection_frequencies,
    top_l_indices,
    uniform_sphere,
)

__all__ = [
    "combinatorial_codes",
    "multi_restart_recovery",
    "one_atom_iar",
    "orthogonal_dictionary",
    "refine_step",
    "selection_counts",
    "selection_frequencies",
    "top_l_indices",
    "uniform_sphere",
]
