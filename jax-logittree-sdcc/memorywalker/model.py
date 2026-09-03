"""A small decoder-only transformer that takes an arbitrary additive 4D mask.

Two things the paper needs from the backbone (Proposition 4, App. C.2):

* condition (i)/(iii): attention visibility is an additive bias supplied by the
  caller, decoupled from the loss mask, so the packed 4D mask of Eq. 7 is
  expressible;
* condition (ii): position ids are reassigned per query row, so a row and its
  image in the compact branch sequence see the same relative offsets. The proof
  needs ``q_i . k_j`` to depend only on that offset, which is why the positional
  encoding here is relative.

So attention takes an integer matrix ``delta[i, j]`` of relative offsets instead
of a vector of position ids. Scores are

    s_ij = sum_m A_ijm cos(theta_m delta_ij) + B_ijm sin(theta_m delta_ij),
    A_ijm = q_i,2m k_j,2m + q_i,2m+1 k_j,2m+1,
    B_ijm = q_i,2m k_j,2m+1 - q_i,2m+1 k_j,2m,

which is what RoPE computes when ``delta_ij = j - i``, written so a row can use
its own gap-free numbering. Condition (iv) holds because the only reductions
across positions are the attention softmax itself; RMSNorm and the MLP are
per-token.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# Masked entries use a large finite negative value rather than -inf: exp of it
# underflows to exactly zero in float64, and no inf ever reaches the backward pass.
NEG = -1e30

Params = dict[str, Any]


class Config(NamedTuple):
    """Hashable so it can be a static argument to the jitted forward."""

    vocab: int = 23
    d_model: int = 24
    n_heads: int = 3
    n_layers: int = 2
    d_ff: int = 48
    rope_base: float = 100.0


def default_config(**kw) -> Config:
    cfg = Config(**kw)
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must divide evenly across heads")
    if (cfg.d_model // cfg.n_heads) % 2:
        raise ValueError("head dimension must be even for the rotary pairing")
    return cfg


def init_params(key: jax.Array, cfg: Config) -> Params:
    d, f, v = cfg.d_model, cfg.d_ff, cfg.vocab
    keys = jax.random.split(key, 2 + 7 * cfg.n_layers)
    it = iter(keys)

    def normal(k, shape, scale):
        return jax.random.normal(k, shape, dtype=jnp.float64) * scale

    params: Params = {
        "embed": normal(next(it), (v, d), 0.5),
        "unembed": normal(next(it), (d, v), 0.5 / d**0.5),
        "blocks": [],
    }
    for _ in range(cfg.n_layers):
        params["blocks"].append({
            "ln1": jnp.ones((d,), dtype=jnp.float64) + normal(next(it), (d,), 0.02),
            "wq": normal(next(it), (d, d), d ** -0.5),
            "wk": normal(next(it), (d, d), d ** -0.5),
            "wv": normal(next(it), (d, d), d ** -0.5),
            "wo": normal(next(it), (d, d), d ** -0.5),
            "ln2": jnp.ones((d,), dtype=jnp.float64),
            "w1": normal(next(it), (d, f), d ** -0.5),
            "w2": normal(next(it), (f, d), f ** -0.5),
        })
    return params


def _rms_norm(x: jnp.ndarray, gain: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * gain


def _attention(block: Params, x: jnp.ndarray, bias: jnp.ndarray,
               delta: jnp.ndarray, n_heads: int, rope_base: float) -> jnp.ndarray:
    n, d = x.shape
    dh = d // n_heads
    q = (x @ block["wq"]).reshape(n, n_heads, dh).transpose(1, 0, 2)
    k = (x @ block["wk"]).reshape(n, n_heads, dh).transpose(1, 0, 2)
    v = (x @ block["wv"]).reshape(n, n_heads, dh).transpose(1, 0, 2)

    m = dh // 2
    inv_freq = rope_base ** (-jnp.arange(m, dtype=jnp.float64) / m)
    angle = delta.astype(jnp.float64)[:, :, None] * inv_freq[None, None, :]  # [n, n, m]
    cos, sin = jnp.cos(angle), jnp.sin(angle)

    qe, qo = q[..., 0::2], q[..., 1::2]
    ke, ko = k[..., 0::2], k[..., 1::2]
    a = jnp.einsum("him,hjm->hijm", qe, ke) + jnp.einsum("him,hjm->hijm", qo, ko)
    b = jnp.einsum("him,hjm->hijm", qe, ko) - jnp.einsum("him,hjm->hijm", qo, ke)
    scores = jnp.sum(a * cos[None] + b * sin[None], axis=-1) / jnp.sqrt(dh)

    scores = scores + bias
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hij,hjd->hid", weights, v)
    return out.transpose(1, 0, 2).reshape(n, d) @ block["wo"]


@functools.partial(jax.jit, static_argnums=(4,))
def forward(params: Params, tokens: jnp.ndarray, bias: jnp.ndarray,
            delta: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Logits for one packed sequence.

    ``bias`` is ``[n, n]`` or ``[n_heads, n, n]`` and is added to the attention
    scores; ``delta[i, j]`` is the relative offset row ``i`` assigns to key ``j``.
    """
    if bias.ndim == 2:
        bias = bias[None]
    x = params["embed"][tokens]
    for block in params["blocks"]:
        x = x + _attention(block, _rms_norm(x, block["ln1"]), bias, delta,
                           cfg.n_heads, cfg.rope_base)
        h = _rms_norm(x, block["ln2"])
        x = x + jax.nn.gelu(h @ block["w1"]) @ block["w2"]
    return x @ params["unembed"]


def causal_bias(n: int) -> jnp.ndarray:
    i = jnp.arange(n)
    return jnp.where(i[:, None] >= i[None, :], 0.0, NEG)


def causal_delta(n: int) -> jnp.ndarray:
    i = jnp.arange(n)
    return i[None, :] - i[:, None]


def sequence_logits(params: Params, token_ids, cfg: Config) -> jnp.ndarray:
    """Ordinary causal forward over a plain sequence of token ids."""
    tokens = jnp.asarray(token_ids)
    n = tokens.shape[0]
    return forward(params, tokens, causal_bias(n), causal_delta(n), cfg)


def cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Summed negative log-likelihood. Every method here sums, never averages,
    so losses from different materializations are comparable term by term."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.take_along_axis(logp, targets[:, None], axis=-1))
