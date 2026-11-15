"""One-layer softmax attention ATC model (paper §3.1, eqs 1–9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ATCConfig:
    """Architecture flags. Defaults match the theoretical model of §3.1."""

    d: int = 16
    n: int = 6
    max_phases: int = 8
    use_mlp: bool = False
    use_layernorm: bool = False
    use_residual: bool = False
    use_positional: bool = False
    mlp_mult: int = 4
    # How to read the next continuous CoT token after a phase succeeds.
    # "readout" = last-layer hidden at z_final (f_tf of §2).
    # "xn_or_last_cot" = hidden at x_n in phase 1, else at the last CoT position (§3).
    extract: str = "readout"

    @staticmethod
    def tiny(n: int = 4, d: int = 32) -> "ATCConfig":
        """1-layer 1-head transformer used in the tiny training test (paper uses d=64)."""
        return ATCConfig(
            d=d,
            n=n,
            max_phases=n,
            use_mlp=True,
            use_layernorm=True,
            use_residual=True,
            use_positional=True,
            extract="readout",
        )


class ForwardResult(NamedTuple):
    yhat: jnp.ndarray
    outputs: jnp.ndarray
    attn: jnp.ndarray
    embeddings: jnp.ndarray


def _glorot(key, shape):
    fan_in = shape[0]
    fan_out = shape[-1]
    std = jnp.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape) * std


def init_params(key, config: ATCConfig):
    """Random parameters for eqs 1–9, plus optional transformer extras."""
    d = config.d
    keys = jax.random.split(key, 12)
    params = {
        "E": _glorot(keys[0], (2, d)),
        "z_final": jax.random.normal(keys[1], (d,)) * (1.0 / jnp.sqrt(d)),
        "WQ": _glorot(keys[2], (d, d)),
        "WK": _glorot(keys[3], (d, d)),
        "WV": _glorot(keys[4], (d, d)),
        "W_heads": jax.random.normal(keys[5], (config.max_phases, d))
        * (1.0 / jnp.sqrt(d)),
    }
    max_len = config.n + max(config.max_phases - 1, 0) + 1
    if config.use_positional:
        params["P"] = jax.random.normal(keys[6], (max_len, d)) * (1.0 / jnp.sqrt(d))
    if config.use_layernorm:
        params["ln1_g"] = jnp.ones((d,))
        params["ln1_b"] = jnp.zeros((d,))
        params["ln2_g"] = jnp.ones((d,))
        params["ln2_b"] = jnp.zeros((d,))
    if config.use_mlp:
        hid = config.mlp_mult * d
        params["W1"] = _glorot(keys[7], (d, hid))
        params["b1"] = jnp.zeros((hid,))
        params["W2"] = _glorot(keys[8], (hid, d))
        params["b2"] = jnp.zeros((d,))
    if config.use_residual or config.use_mlp:
        params["WO"] = _glorot(keys[9], (d, d))
    return params


def empty_cots(batch_size: int, config: ATCConfig):
    return jnp.zeros((batch_size, 0, config.d), dtype=jnp.float32)


def _layernorm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) * jax.lax.rsqrt(var + eps) + beta


def assemble_embeddings(params, bits, cots, config: ATCConfig):
    """Token embeddings: word table on ±1 bits (eq 1), CoT skip-embed (eq 2), z_final (eq 3)."""
    bits = jnp.asarray(bits, dtype=jnp.float32)
    if bits.ndim == 1:
        bits = bits[None, :]
        squeeze = True
    else:
        squeeze = False
    if cots is None:
        cots = empty_cots(bits.shape[0], config)
    else:
        cots = jnp.asarray(cots, dtype=jnp.float32)
        if cots.ndim == 2:
            cots = cots[None, :, :]
    idx = jnp.astype(bits < 0, jnp.int32)
    e_bits = params["E"][idx]
    z_final = jnp.broadcast_to(params["z_final"], (bits.shape[0], 1, config.d))
    tokens = jnp.concatenate([e_bits, cots, z_final], axis=1)
    if config.use_positional:
        tokens = tokens + params["P"][: tokens.shape[1]]
    if squeeze:
        tokens = tokens[0]
    return tokens


def causal_attention(q, k, v):
    """Scaled dot-product attention with a causal mask (eqs 7–8).

    ``q, k, v`` are ``(..., L, d)``. Returns ``(o, alpha)`` with ``alpha[..., i, j] = 0``
    for ``j > i``, and each row a probability vector.
    """
    d_k = k.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(d_k, q.dtype))
    L = scores.shape[-1]
    causal = jnp.tril(jnp.ones((L, L), dtype=bool))
    scores = jnp.where(causal, scores, jnp.asarray(-1e9, dtype=scores.dtype))
    alpha = jax.nn.softmax(scores, axis=-1)
    o = jnp.matmul(alpha, v)
    return o, alpha


def _block(params, tokens, config: ATCConfig):
    """One-layer attention (± optional LN / residual / MLP). Default is eqs 4–8."""
    x = tokens
    if config.use_layernorm:
        x = _layernorm(x, params["ln1_g"], params["ln1_b"])
    q = x @ params["WQ"]
    k = x @ params["WK"]
    v = x @ params["WV"]
    attn_out, alpha = causal_attention(q, k, v)
    if "WO" in params:
        attn_out = attn_out @ params["WO"]
    if config.use_residual:
        attn_out = attn_out + tokens
    if config.use_mlp:
        h = attn_out
        if config.use_layernorm:
            h = _layernorm(h, params["ln2_g"], params["ln2_b"])
        mlp_out = jax.nn.gelu(h @ params["W1"] + params["b1"]) @ params["W2"] + params[
            "b2"
        ]
        if config.use_residual:
            outputs = attn_out + mlp_out
        else:
            outputs = mlp_out
    else:
        outputs = attn_out
    return outputs, alpha


def forward(params, bits, cots, phase: int, config: ATCConfig, head_index=None):
    """Run the model and predict from the final-answer position (eq 9).

    Phase ``r`` uses ``r-1`` CoT tokens plus ``z_final`` (§3). ``head_index``
    overrides the per-phase linear head (used by the data-curriculum-only baseline).
    """
    bits = jnp.asarray(bits, dtype=jnp.float32)
    squeeze = bits.ndim == 1
    if squeeze:
        bits = bits[None, :]
        if cots is not None and jnp.asarray(cots).ndim == 2:
            cots = jnp.asarray(cots)[None, :, :]
    embeddings = assemble_embeddings(params, bits, cots, config)
    outputs, alpha = _block(params, embeddings, config)
    idx = head_index if head_index is not None else (phase - 1)
    w = params["W_heads"][idx]
    yhat = jnp.einsum("bd,d->b", outputs[:, -1, :], w)
    if squeeze:
        return ForwardResult(yhat[0], outputs[0], alpha[0], embeddings[0])
    return ForwardResult(yhat, outputs, alpha, embeddings)


def extract_index(n_cots: int, config: ATCConfig) -> int:
    """Position whose hidden state becomes the next CoT token."""
    if config.extract == "xn_or_last_cot":
        if n_cots == 0:
            return config.n - 1
        return config.n + n_cots - 1
    return -1


def build_cots(params, bits, n_cots: int, config: ATCConfig):
    """Recursively form z_1..z_{n_cots} with identity T_i and stop-gradient.

    Each new token is the last-layer hidden at the extraction position after a
    forward with the CoTs already produced. Gradients do not flow into earlier
    thoughts (truncated to the current answer head, as in the cheap ATC loop).
    """
    bits = jnp.asarray(bits, dtype=jnp.float32)
    squeeze = bits.ndim == 1
    if squeeze:
        bits = bits[None, :]
    B = bits.shape[0]
    cots = empty_cots(B, config)
    for i in range(n_cots):
        res = forward(params, bits, cots, phase=i + 1, config=config)
        pos = extract_index(i, config)
        z = res.outputs[:, pos, :]
        z = jax.lax.stop_gradient(z)
        cots = jnp.concatenate([cots, z[:, None, :]], axis=1)
    if squeeze:
        return cots[0]
    return cots


def mse_loss(params, bits, cots, y, phase: int, config: ATCConfig, head_index=None):
    res = forward(params, bits, cots, phase, config, head_index=head_index)
    y = jnp.asarray(y, dtype=jnp.float32)
    return jnp.mean((res.yhat - y) ** 2)


def sign_accuracy(yhat, y):
    yhat = jnp.asarray(yhat)
    y = jnp.asarray(y)
    return jnp.mean(jnp.sign(yhat) == jnp.sign(y))
