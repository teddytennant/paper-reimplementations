"""JAX implementation of the three algorithms in "Kernel Reboot: Breaking the
Boundaries of Neural Tangent Kernels for Neural Fields" (arXiv:2609.03117).

Everything here runs in float64. Kernel matrices and their solves are the part of
this pipeline that is sensitive to precision, and a TF32 matmul on an H200 is not
accurate enough to keep two nearby reconstructions in the right order.
"""

import jax

jax.config.update("jax_enable_x64", True)

from metaquill_kip.field import (  # noqa: E402
    MLPConfig,
    encode,
    fourier_pe,
    init_params,
    mlp_apply,
    raw_pe,
)
from metaquill_kip.kip import (  # noqa: E402
    kip_loss,
    kip_predict,
    ntk_kip,
)
from metaquill_kip.metaquill import (  # noqa: E402
    kernel_coefficients,
    lift_to_weights,
    meta_train,
    metaquill_adapt,
    tangent_predict,
)
from metaquill_kip.metaquill_kip import metaquill_kip_adapt  # noqa: E402
from metaquill_kip.metrics import psnr  # noqa: E402
from metaquill_kip.ntk import (  # noqa: E402
    jacobian_matrix,
    kernel_predict,
    kernel_solve,
    ntk_matrix,
    ntk_matrix_jvp,
    ntk_regression,
)

__all__ = [
    "MLPConfig",
    "encode",
    "fourier_pe",
    "init_params",
    "jacobian_matrix",
    "kernel_coefficients",
    "kernel_predict",
    "kernel_solve",
    "kip_loss",
    "kip_predict",
    "lift_to_weights",
    "meta_train",
    "metaquill_adapt",
    "metaquill_kip_adapt",
    "mlp_apply",
    "ntk_kip",
    "ntk_matrix",
    "ntk_matrix_jvp",
    "ntk_regression",
    "psnr",
    "raw_pe",
    "tangent_predict",
]
