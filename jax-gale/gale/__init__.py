"""GaLe: memory-efficient Global Approximate and Local Exact features (arXiv:2609.02689)."""

from .attention import (
    attention,
    attention_map,
    effective_attention_matrix,
    full_map_elems,
    gale_attention,
)
from .calibrate import Calibration, calibrate, overlap_vs_ppbi
from .fusion import (
    ALPHA_DEFAULT,
    ALPHA_STABLE_RANGE,
    bilinear_down,
    bilinear_up,
    fuse,
    ga_branch,
    gale_forward,
    mse,
)
from .layers import Conv, GlobalGate, MaxPool, Relu, exact_forward, make_stack, tensor_shapes
from .receptive_field import (
    calibration_step,
    ppbi_overlap,
    receptive_field,
    suffix_receptive_fields,
)
from .slicing import (
    overlap_caps,
    overlap_profile,
    BlockPlan,
    MemoryTrace,
    bands,
    exact_forward_trace,
    plan_adaptive,
    run_plan,
    sliced_forward,
    snap_halos,
)

__all__ = [
    "ALPHA_DEFAULT",
    "ALPHA_STABLE_RANGE",
    "BlockPlan",
    "Calibration",
    "Conv",
    "GlobalGate",
    "MaxPool",
    "MemoryTrace",
    "Relu",
    "attention",
    "attention_map",
    "bands",
    "bilinear_down",
    "bilinear_up",
    "calibrate",
    "calibration_step",
    "effective_attention_matrix",
    "exact_forward",
    "exact_forward_trace",
    "full_map_elems",
    "overlap_caps",
    "overlap_profile",
    "fuse",
    "ga_branch",
    "gale_attention",
    "gale_forward",
    "make_stack",
    "mse",
    "overlap_vs_ppbi",
    "plan_adaptive",
    "ppbi_overlap",
    "receptive_field",
    "run_plan",
    "sliced_forward",
    "snap_halos",
    "suffix_receptive_fields",
    "tensor_shapes",
]
