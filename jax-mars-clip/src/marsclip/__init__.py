"""JAX implementation of MARS-CLIP (arXiv:2609.08283).

Training-free CLIP surgery on arrays: multi-resolution fusion (Eq. 1),
structure-aware KK^T attention with spatial/color biases (Eq. 2--3),
and cosine zero-shot scores (Eq. 4). No CLIP checkpoints or datasets.
"""

from marsclip.attention import (
    combine_biases,
    key_key_logits,
    linear,
    refined_attention_block,
    structure_aware_attention,
)
from marsclip.color import (
    COLOR_SIGMA,
    color_affinity,
    color_bias,
    patch_lab_colors,
    srgb_to_lab,
)
from marsclip.constants import ALPHA, ENCODER_SIZE, S_CROP
from marsclip.fusion import (
    encode_multiresolution,
    fuse_features,
    grid_crops,
    pad_to_multiple,
    reassemble_local_features,
    resize_chw,
    upsample_features,
)
from marsclip.scores import cosine_scores, predict_mask

__version__ = "0.1.0"

__all__ = [
    "ALPHA",
    "COLOR_SIGMA",
    "ENCODER_SIZE",
    "S_CROP",
    "color_affinity",
    "color_bias",
    "combine_biases",
    "cosine_scores",
    "encode_multiresolution",
    "fuse_features",
    "grid_crops",
    "key_key_logits",
    "linear",
    "pad_to_multiple",
    "patch_lab_colors",
    "predict_mask",
    "reassemble_local_features",
    "refined_attention_block",
    "resize_chw",
    "srgb_to_lab",
    "structure_aware_attention",
    "upsample_features",
]
