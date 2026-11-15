"""MercerFlow: flow matching in the Mercer eigenbasis of a GP prior.

Implements the model in arXiv:2610.06039.
"""

from mercerflow.codecs import (
    build_basis,
    dct_basis,
    decode,
    encode,
    fix_signs,
    fourier_basis,
    mercer_basis,
    pca_basis,
    raw_basis,
)
from mercerflow.diagnostics import (
    jacobi_condition_number,
    jacobi_scaled,
    prior_coefficient_covariance,
)
from mercerflow.flow import (
    encode_endpoints,
    euler_forecast,
    euler_integrate,
    flow_matching_loss,
    interpolant,
    sample_cfm_batch,
    shared_context_velocity,
    train_step,
    training_step,
)
from mercerflow.kernels import (
    conditional_moments,
    gram_matrix,
    ou_kernel,
    partition_gram,
    sample_conditional_future,
    sample_prior,
)
from mercerflow.model import (
    apply_velocity_mlp,
    count_parameters,
    init_velocity_mlp,
    sinusoidal_time_embedding,
)

__all__ = [
    "apply_velocity_mlp",
    "build_basis",
    "conditional_moments",
    "count_parameters",
    "dct_basis",
    "decode",
    "encode",
    "encode_endpoints",
    "euler_forecast",
    "euler_integrate",
    "fix_signs",
    "flow_matching_loss",
    "fourier_basis",
    "gram_matrix",
    "init_velocity_mlp",
    "interpolant",
    "jacobi_condition_number",
    "jacobi_scaled",
    "mercer_basis",
    "ou_kernel",
    "partition_gram",
    "pca_basis",
    "prior_coefficient_covariance",
    "raw_basis",
    "sample_cfm_batch",
    "sample_conditional_future",
    "sample_prior",
    "shared_context_velocity",
    "sinusoidal_time_embedding",
    "train_step",
    "training_step",
]
