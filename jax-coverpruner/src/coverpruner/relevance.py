"""Query demand from the layer-0 attention probe: Eq. 4 of arXiv:2609.03158.

The demand weight w_i says how much the current instruction wants the model to
consult visual token i. The paper reads it from the first decoder layer only,
before any pruning happens, using the instruction positions T_q as queries and the
visual-token positions as keys. It is a probe, not an attention map harvested from
a full decoder pass.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["layer0_attention_probe", "relevance_from_probe", "demand_weights"]


def _as_float(x):
    return jnp.asarray(x, dtype=jnp.result_type(float))


def layer0_attention_probe(
    h_instruction,
    h_visual,
    w_q,
    w_k,
    num_heads,
    *,
    ln_scale=None,
    ln_bias=None,
    ln_eps=1e-5,
):
    """Scaled dot-product attention alpha^(0,h)_{t,i} from layer-0 weights (Eq. 4).

    Args:
        h_instruction: (T_q, d) decoder input states at the instruction positions.
        h_visual: (n, d) decoder input states at the visual-token positions, i.e. the
            projected tokens sitting in the model's native input layout.
        w_q, w_k: (d, num_heads * d_head) layer-0 query and key projections.
        num_heads: H.
        ln_scale, ln_bias: layer-0 input LayerNorm parameters, applied to both sides
            when given. RMS-norm models pass ln_bias=None.
        ln_eps: LayerNorm epsilon.

    Returns:
        (H, T_q, n) attention weights.

    The softmax runs over the visual keys only. A real decoder softmaxes each query
    over its whole causal prefix, so this normalizes the visual share of that mass to
    one; the paper does not say which it uses, and it only matters up to a per-row
    scale that min-max normalization in demand_weights partly absorbs anyway. Rotary
    or learned position terms are not applied here.
    """
    hq = _layer_norm(_as_float(h_instruction), ln_scale, ln_bias, ln_eps)
    hv = _layer_norm(_as_float(h_visual), ln_scale, ln_bias, ln_eps)
    w_q, w_k = _as_float(w_q), _as_float(w_k)

    t_q, n = hq.shape[0], hv.shape[0]
    d_head = w_q.shape[1] // num_heads
    q = jnp.matmul(hq, w_q, precision="highest").reshape(t_q, num_heads, d_head)
    k = jnp.matmul(hv, w_k, precision="highest").reshape(n, num_heads, d_head)

    scores = jnp.einsum("thc,nhc->htn", q, k, precision="highest")
    return _softmax(scores / jnp.sqrt(jnp.asarray(d_head, dtype=scores.dtype)))


def _layer_norm(x, scale, bias, eps):
    if scale is None and bias is None:
        return x
    x = x - jnp.mean(x, axis=-1, keepdims=True)
    x = x / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + eps)
    if scale is not None:
        x = x * _as_float(scale)
    if bias is not None:
        x = x + _as_float(bias)
    return x


def _softmax(x):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def relevance_from_probe(alpha):
    """r_i = (1 / (|T_q| H)) sum_t sum_h alpha^(0,h)_{t,i} (Eq. 4).

    Args:
        alpha: (..., n) probe attention, with any number of leading axes to average
            over. The natural layout is (H, T_q, n) as returned by
            layer0_attention_probe.

    Returns:
        (n,) raw relevance, before normalization.
    """
    alpha = _as_float(alpha)
    return jnp.mean(alpha.reshape(-1, alpha.shape[-1]), axis=0)


def demand_weights(r, tau=0.3):
    """w_i = exp(tau * (r_i - r_min) / (r_max - r_min)), Sec. 3.3 after Eq. 4.

    Args:
        r: (n,) raw relevance from relevance_from_probe.
        tau: temperature. tau = 0 gives w_i = 1 for all i, which is unweighted
            coverage; larger tau steers budget toward query-relevant tokens. The
            paper's default is 0.3 and its Table 9 sweep is flat around it.

    Returns:
        (n,) weights in [1, e^tau], so w >= 0 as Eq. 2 requires.

    A constant r (r_max = r_min) falls back to uniform weights instead of dividing
    by zero.
    """
    r = _as_float(r)
    lo, hi = jnp.min(r), jnp.max(r)
    span = hi - lo
    r_tilde = jnp.where(span > 0, (r - lo) / jnp.where(span > 0, span, 1.0), 0.0)
    return jnp.exp(_as_float(tau) * r_tilde)
