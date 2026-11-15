import os

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

from tsgate.gate import (
    apply_output_gate,
    gate_values,
    gate_width,
    image_to_text_fraction,
    init_stream_params,
    joint_attention,
    peak_attention_mass,
    project_q_and_content,
    rmsnorm,
    rope,
    silu,
    time_bias,
    time_branch_param_count,
)

__all__ = [
    "apply_output_gate",
    "gate_values",
    "gate_width",
    "image_to_text_fraction",
    "init_stream_params",
    "joint_attention",
    "peak_attention_mass",
    "project_q_and_content",
    "rmsnorm",
    "rope",
    "silu",
    "time_bias",
    "time_branch_param_count",
]
