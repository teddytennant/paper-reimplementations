"""Tiny tensor residual-attention fusion gates (PRI-Net eqs. 9-11).

No CNN / MLP backbone. Eq. 9 is the sigmoid gate; the paper's h_mlp is
omitted and the caller passes logits (or precomputed weights). Eq. 10 is the
residual feature gate. Eq. 11 is residual regression, with an optional
multiplicative gate on Delta y so that gate=0 leaves the coarse image-branch
estimate unchanged and gate=1 recovers the paper formula.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def attention_weights(logits: Array) -> Array:
    """Modality gate w^m = sigma(logits) (eq. 9 without the MLP)."""
    return jax.nn.sigmoid(logits)


def residual_gate(x: Array, w: Array) -> Array:
    """Residual feature update x̃ = x + x ⊙ w (eq. 10)."""
    return x + x * w


def fuse_features(x_c: Array, x_l: Array, w_c: Array, w_l: Array) -> Array:
    """Gated residual features and concat x^n = [x̃^c ; x̃^l] (eqs. 10-11)."""
    x_c_tilde = residual_gate(x_c, w_c)
    x_l_tilde = residual_gate(x_l, w_l)
    return jnp.concatenate([x_c_tilde, x_l_tilde], axis=-1)


def residual_regression(
    y_hat_c: Array,
    delta_y: Array,
    gate: Array | float = 1.0,
) -> Array:
    """Image-branch coarse estimate plus residual (eq. 11).

    Paper: ŷ = ŷ^c + Δy = H_c(x^c) + H_F(x^n).
    The optional gate yields ŷ = ŷ^c + gate ⊙ Δy. Default gate=1 matches
    the paper. gate=0 leaves the coarse estimate unchanged.
    """
    return y_hat_c + gate * delta_y
