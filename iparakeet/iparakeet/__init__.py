"""Integer relative-attention score fusion and a minimax Swish approximation.

Kernels follow arXiv 2609.30846. Softmax stays in float; this package does not
rederive the on-device I-BERT softmax.
"""

from iparakeet.attention import (
    activation_scales,
    float_attention,
    fuse_scores_dyadic,
    fuse_scores_reference,
    integer_attention,
    phi,
    rel_shift,
)
from iparakeet.range import dequantize, percentile_scale, quantize
from iparakeet.swish import SWISH_A, SWISH_C, hard_swish, swish, swish_hat, tanh_hat

__all__ = [
    "SWISH_A",
    "SWISH_C",
    "activation_scales",
    "dequantize",
    "float_attention",
    "fuse_scores_dyadic",
    "fuse_scores_reference",
    "hard_swish",
    "integer_attention",
    "percentile_scale",
    "phi",
    "quantize",
    "rel_shift",
    "swish",
    "swish_hat",
    "tanh_hat",
]
