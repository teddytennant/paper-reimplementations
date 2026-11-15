"""CADMP: cross-modal attention drift plus mask-based verification.

Implements the method of arXiv:2609.02028 with the LVLM treated as an input.
Start at ``cadmp.features.extract``.
"""

from .attention import attention_distributions, head_average, normalize
from .drift import drift_profile, kl_divergence, select_mask_layer, smooth
from .features import CADMPFeatures, extract, extract_drift, feature_dim
from .masking import GRAY, apply_mask, attention_mask, mask_image, upsample_attention
from .detector import Detector, accuracy, auroc, predict, train

__all__ = [
    "attention_distributions",
    "head_average",
    "normalize",
    "drift_profile",
    "kl_divergence",
    "select_mask_layer",
    "smooth",
    "CADMPFeatures",
    "extract",
    "extract_drift",
    "feature_dim",
    "GRAY",
    "apply_mask",
    "attention_mask",
    "mask_image",
    "upsample_attention",
    "Detector",
    "accuracy",
    "auroc",
    "predict",
    "train",
]
