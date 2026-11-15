"""A tiny causal transformer, written functionally so params are a plain dict.

This stands in for the Qwen3 policy pi_theta of the paper. It is deliberately
small enough that the whole trajectory space can be enumerated in a test, which
is how test_control_variate checks E_{o~pi}[grad log pi(o|q)] = 0 exactly rather
than by Monte Carlo.

Everything takes an explicit dtype so the exactness tests can run in float64.
On NVIDIA, float32 matmuls default to TF32, and a finite-difference gradient
check does not survive that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = ["ModelConfig", "init_params", "forward", "log_probs"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_len: int
    d_model: int = 32
    n_heads: int = 2
    d_ff: int = 64

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


def init_params(key, config: ModelConfig, dtype=jnp.float32):
    """Small random init. One attention block, one MLP, no layer norm."""
    keys = jr.split(key, 8)
    d, f, v, t = config.d_model, config.d_ff, config.vocab_size, config.max_len
    s = 1.0 / math.sqrt(d)
    return {
        "tok_emb": jr.normal(keys[0], (v, d), dtype) * s,
        "pos_emb": jr.normal(keys[1], (t, d), dtype) * s,
        "wq": jr.normal(keys[2], (d, d), dtype) * s,
        "wk": jr.normal(keys[3], (d, d), dtype) * s,
        "wv": jr.normal(keys[4], (d, d), dtype) * s,
        "wo": jr.normal(keys[5], (d, d), dtype) * s,
        "w1": jr.normal(keys[6], (d, f), dtype) * s,
        "b1": jnp.zeros((f,), dtype),
        "w2": jr.normal(keys[7], (f, d), dtype) * (1.0 / math.sqrt(f)),
        "b2": jnp.zeros((d,), dtype),
        "out": jr.normal(keys[0], (d, v), dtype) * s,
    }


def forward(params, config: ModelConfig, tokens):
    """Logits for one sequence. tokens is (T,) int, output is (T, vocab)."""
    seq_len = tokens.shape[0]
    h = params["tok_emb"][tokens] + params["pos_emb"][:seq_len]
    d = config.d_model
    heads, head_dim = config.n_heads, d // config.n_heads

    q = (h @ params["wq"]).reshape(seq_len, heads, head_dim)
    k = (h @ params["wk"]).reshape(seq_len, heads, head_dim)
    v = (h @ params["wv"]).reshape(seq_len, heads, head_dim)
    scores = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(head_dim)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    attn = jax.nn.softmax(scores, axis=-1)
    h = h + jnp.einsum("hqk,khd->qhd", attn, v).reshape(seq_len, d) @ params["wo"]
    h = h + jax.nn.gelu(h @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]
    return h @ params["out"]


def log_probs(params, config: ModelConfig, tokens, temperature=1.0):
    """Per-position next-token log probabilities, shifted so index i scores tokens[i+1].

    Returns (T-1,). Temperature matters here because p_s(q) in the paper is the
    success rate of the sampler actually used for rollouts, which the paper runs
    at T = 1.0.
    """
    logits = forward(params, config, tokens)[:-1] / temperature
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, tokens[1:, None], axis=-1)[:, 0]
