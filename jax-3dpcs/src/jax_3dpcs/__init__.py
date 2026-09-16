"""JAX 3D point-cloud splatting (3DPCS) from PRI-Net, arXiv:2609.14469."""

from jax_3dpcs.camera import camera_matrix, project_P, project_pinhole
from jax_3dpcs.raf import attention_weights, residual_gate, residual_regression
from jax_3dpcs.splat import SplatMaps, splat_3dpcs, splat_projected

__all__ = [
    "SplatMaps",
    "attention_weights",
    "camera_matrix",
    "project_P",
    "project_pinhole",
    "residual_gate",
    "residual_regression",
    "splat_3dpcs",
    "splat_projected",
]

__version__ = "0.1.0"
