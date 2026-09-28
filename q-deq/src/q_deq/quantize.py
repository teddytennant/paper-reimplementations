"""W8A8 fake quantization: eq 7 (per-channel weights), eq 8 (per-tensor acts), STE."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from q_deq.deq import shared_layer

QMIN = -127.0
QMAX = 127.0


@jax.custom_jvp
def ste_round(x):
    """Round with straight-through estimator (identity JVP)."""
    return jnp.round(x)


@ste_round.defjvp
def _ste_round_jvp(primals, tangents):
    (x,) = primals
    (tx,) = tangents
    return jnp.round(x), tx


def weight_scale(W, eps=1e-8):
    """Eq 7: s_{w,c} = max(|W_{:,c}|) / 127  (per column / channel)."""
    s = jnp.max(jnp.abs(W), axis=0) / 127.0
    return jnp.maximum(s, eps)


def activation_scale(A, eps=1e-8):
    """Eq 8: s_a = max(|A|) / 127  (per tensor)."""
    s = jnp.max(jnp.abs(A)) / 127.0
    return jnp.maximum(s, eps)


def quantize_with_scale(X, s):
    """s * clip(round(X / s), -127, 127). STE on round; clip zeros outside."""
    s = jnp.asarray(s, dtype=X.dtype)
    q = jnp.clip(ste_round(X / s), QMIN, QMAX)
    return s * q


def quantize_weight(W, eps=1e-8):
    """Per-channel weight fake-quant (eq 7). W is (K, C); channel = column."""
    s = weight_scale(W, eps=eps)
    return quantize_with_scale(W, s), s


def quantize_activation(A, eps=1e-8):
    """Per-tensor activation fake-quant (eq 8)."""
    s = activation_scale(A, eps=eps)
    return quantize_with_scale(A, s), s


def _q_w(W):
    Wq, _ = quantize_weight(W)
    return Wq


def _q_a(A):
    Aq, _ = quantize_activation(A)
    return Aq


def quantized_shared_layer(params, z, x):
    """Re-forward the shared layer under W8A8 fake quant of weights and acts."""
    qparams = {
        "Wz": _q_w(params["Wz"]),
        "Wx": _q_w(params["Wx"]),
        "b": params["b"],
        "W1": _q_w(params["W1"]),
        "b1": params["b1"],
        "W2": _q_w(params["W2"]),
        "b2": params["b2"],
    }
    z_q = _q_a(z)
    x_q = _q_a(x)
    return shared_layer(qparams, z_q, x_q)
