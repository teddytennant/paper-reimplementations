"""Learnable orthogonal subspace projections for frozen linear layers."""

from learnable_subspace.allocate import (
    allocate,
    monotone_running_max,
    savings,
    untied_c,
    untied_unit,
)
from learnable_subspace.init import (
    WhitenedInit,
    cholesky_covariance,
    empirical_gram,
    output_side_init,
    reconstruction_energy,
    tail_energy,
    whitened_input_init,
)
from learnable_subspace.merge import (
    merge_input,
    merge_output,
    merge_tied_input,
    orthogonal_complement,
)
from learnable_subspace.projector import (
    dense_projector,
    forward_input,
    forward_output,
    orthogonality_penalty,
    orthogonal_factor,
    project_activations,
    sample_direction_mask,
    warmup_alpha,
)
from learnable_subspace.train import TwoLayerMLP, fit_mlp, init_mlp, softmax_kl, subspace_loss

__all__ = [
    "TwoLayerMLP",
    "WhitenedInit",
    "allocate",
    "cholesky_covariance",
    "dense_projector",
    "empirical_gram",
    "fit_mlp",
    "forward_input",
    "forward_output",
    "init_mlp",
    "merge_input",
    "merge_output",
    "merge_tied_input",
    "monotone_running_max",
    "orthogonal_complement",
    "orthogonal_factor",
    "orthogonality_penalty",
    "output_side_init",
    "project_activations",
    "reconstruction_energy",
    "sample_direction_mask",
    "savings",
    "softmax_kl",
    "subspace_loss",
    "tail_energy",
    "untied_c",
    "untied_unit",
    "warmup_alpha",
    "whitened_input_init",
]
