"""The CADMP feature vector, and the entry point for a real LVLM.

The detector's input is the drift profile concatenated with three scalars:

    x = [ F_drift , P_orig , P_masked , dP ],     dP = P_orig - P_masked,

with ``F_drift = [K_2, ..., K_L]``. For LLaVA-1.5-7B's 32 decoder layers that is
31 + 3 = 34 dimensions. ``L`` is a parameter everywhere; nothing here assumes 32.

``dP`` is redundant given the first two, in the sense that any linear map of the
34-dim vector could form it. It is fed in anyway because the paper feeds it in,
and because the first layer is followed by a nonlinearity, so handing the
difference over directly is not the same model as letting the network find it.

Sign convention on ``dP``. The paper's story is that grounded predictions lose
confidence when the attended region is grayed out, so grounded samples have
*large positive* ``dP`` and hallucinated ones have ``dP`` near zero. Positive
means the probability dropped.

Positive class. The appendix says AUROC is computed with hallucinated samples as
the positive class, so ``label = 1`` means hallucinated throughout this package.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .attention import attention_distributions
from .drift import drift_profile, select_mask_layer

__all__ = ["CADMPFeatures", "feature_dim", "extract", "extract_drift"]


class CADMPFeatures(NamedTuple):
    """What one forward pass of CADMP produces for one target token."""

    features: jnp.ndarray
    """``[L + 2]`` detector input, ``[F_drift, P_orig, P_masked, dP]``."""

    drift: jnp.ndarray
    """``[L - 1]`` drift profile ``F_drift``."""

    mask_layer: int
    """0-based index of ``l_mask``, the layer whose attention drove the mask."""


def feature_dim(num_layers: int) -> int:
    """Detector input width for an ``L``-layer model: ``(L - 1) + 3``."""
    return num_layers + 2


def extract_drift(attn: jnp.ndarray, eps: float = 1e-9):
    """Run the attention half of CADMP: raw attention to drift and ``l_mask``.

    Call this first if you are driving a real LVLM, because you need
    ``l_mask`` before you can build the masked image and get ``P_masked``.

    Args:
        attn: ``[L, H, N]`` cross-modal attention from the target-token query
            position to the visual tokens, per layer and head. Raw softmax
            weights; the restriction to the visual subset happens here.
        eps: uniform-mixing weight for the KL, see ``cadmp.drift.smooth``.

    Returns:
        ``(dists, drift, mask_layer)`` where ``dists`` is ``[L, N]``, ``drift``
        is ``[L - 1]``, and ``mask_layer`` is a 0-based layer index. Feed
        ``dists[mask_layer]`` to ``cadmp.masking.mask_image``.
    """
    dists = attention_distributions(attn)
    drift = drift_profile(dists, eps=eps)
    return dists, drift, select_mask_layer(drift)


def extract(
    attn: jnp.ndarray,
    p_orig: float | jnp.ndarray,
    p_masked: float | jnp.ndarray,
    eps: float = 1e-9,
) -> CADMPFeatures:
    """Full CADMP feature extraction for one target token.

    This is the entry point for somebody with an actual LVLM. The usual order of
    operations, with the model held frozen throughout:

    1. Run the model on ``(image, prompt, prefix)``, take the target token ``y``,
       record ``P_orig = P(y | v, t, Y_<t)``, and pull the attention from the
       query position that predicts ``y`` to the ``N`` visual tokens, for every
       layer and head. That is the ``[L, H, N]`` tensor.
    2. ``dists, drift, l = extract_drift(attn)``.
    3. ``masked = cadmp.masking.mask_image(image, dists[l])``.
    4. Run the model again on ``(masked, prompt, prefix)`` and read off
       ``P_masked = P(y | v_masked, t, Y_<t)`` for the *same* token ``y``. Not
       the new argmax, the same token.
    5. Call this function with the attention from step 1 and the two
       probabilities.

    For a multi-token object name the appendix uses the first token, which
    ablates better than the last token or the mean over tokens.

    Args:
        attn: ``[L, H, N]`` cross-modal attention, as in step 1.
        p_orig: ``P_orig``, scalar.
        p_masked: ``P_masked``, scalar.
        eps: uniform-mixing weight for the KL.

    Returns:
        ``CADMPFeatures``. ``.features`` is what the detector consumes.
    """
    _, drift, mask_layer = extract_drift(attn, eps=eps)
    p_orig = jnp.asarray(p_orig, drift.dtype)
    p_masked = jnp.asarray(p_masked, drift.dtype)
    tail = jnp.stack([p_orig, p_masked, p_orig - p_masked])
    return CADMPFeatures(
        features=jnp.concatenate([drift, tail]),
        drift=drift,
        mask_layer=mask_layer,
    )
