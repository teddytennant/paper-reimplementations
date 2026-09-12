"""Axis mixture gates: AXON-Fixed (layer scalar) and AXON-Gated (per token).

AXON-Fixed: lambda = sigmoid(ell), shared by every token in the layer.
AXON-Gated: (alpha, beta) = softmax(g(sg(x_i)) / tau) with a two-layer MLP
g: R^d -> R^2, hidden width d/4, GELU, zero-initialised output. Weights
are non-negative and sum to 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aaa.attention import init_linear


def init_fixed_gate(ell=0.0):
    """Scalar logit; sigmoid(0) = 1/2 so both paths start equal."""
    return {"ell": jnp.asarray(ell, dtype=jnp.float32)}


def init_token_gate(key, d):
    """Two-layer MLP d -> d/4 -> 2 with zero-initialised output layer."""
    hidden = max(d // 4, 1)
    k1, k2 = jax.random.split(key)
    return {
        "W1": init_linear(k1, d, hidden),
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "W2": jnp.zeros((hidden, 2), dtype=jnp.float32),
        "b2": jnp.zeros((2,), dtype=jnp.float32),
    }


def fixed_lambda(params):
    """Layer-wide temporal weight lambda = sigmoid(ell) in (0, 1)."""
    return jax.nn.sigmoid(params["ell"])


def token_gate_weights(params, x, tau=1.0, stop_gradient=True):
    """Per-token (alpha, beta) with alpha + beta = 1.

    x: (B, C, T, D). Returns alpha, beta each (B, C, T, 1), temporal then spatial.
    The MLP sees sg(x) as in Section 2.3.
    """
    h = jax.lax.stop_gradient(x) if stop_gradient else x
    hidden = jax.nn.gelu(h @ params["W1"] + params["b1"])
    logits = (hidden @ params["W2"] + params["b2"]) / tau
    weights = jax.nn.softmax(logits, axis=-1)
    alpha = weights[..., 0:1]
    beta = weights[..., 1:2]
    return alpha, beta


def fixed_mix(params, temporal, spatial):
    """y = lambda T(x) + (1 - lambda) S(x) with one lambda per layer."""
    lam = fixed_lambda(params)
    return lam * temporal + (1.0 - lam) * spatial


def gated_mix(params, x, temporal, spatial, tau=1.0, stop_gradient=True):
    """y_i = alpha_i T(x)_i + beta_i S(x)_i."""
    alpha, beta = token_gate_weights(
        params, x, tau=tau, stop_gradient=stop_gradient
    )
    return alpha * temporal + beta * spatial
