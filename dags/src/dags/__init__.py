"""Control equations for DAGS, arXiv 2610.02567."""

from dags.control import (
    backward_warp,
    demodulate,
    flow_source,
    inject,
    lighting_recurrence,
    remodulate,
    temporal_guidance,
    temporal_lambda,
)
from dags.modules import (
    AppearanceEncoder,
    GeometryEncoder,
    StabilizerStep,
    init_layer_scales,
    inject_layers,
    token_align,
)

__all__ = [
    "AppearanceEncoder",
    "GeometryEncoder",
    "StabilizerStep",
    "backward_warp",
    "demodulate",
    "flow_source",
    "init_layer_scales",
    "inject",
    "inject_layers",
    "lighting_recurrence",
    "remodulate",
    "temporal_guidance",
    "temporal_lambda",
    "token_align",
]

__version__ = "0.1.0"
