"""Two metrics: coordinate-wise K/V R^2, and the receiver-composed attention drift.

Observation O2 in section 1 is that these disagree, so the demo reports both.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .attn_repair import attention
from .rope import apply_rope, rope_angles


def r2(pred, true):
    """Held-out R^2 pooled over layers, heads and coordinates."""
    resid = jnp.sum((pred - true) ** 2)
    total = jnp.sum((true - true.mean(axis=(0, 1), keepdims=True)) ** 2)
    return float(1.0 - resid / total)


def attention_drift(traces, pred_k_content, pred_v, sample_pos, boundaries,
                    q_per_kv: int, head_dim: int):
    """Relative L2 drift of the receiver's attention output when it reads the mapped cache.

    This is the cache-local part of the propagated error in eq 3, measured at the same
    causal boundaries ATTN-REPAIR calibrates on.
    """
    lt, s, t, hq, d = traces.tgt_q_rope.shape
    ht = traces.tgt_k_rope.shape[-2]
    n_pos = sample_pos.shape[0]
    ang = rope_angles(sample_pos, head_dim)[None, None, :, None, :]

    pk = pred_k_content.reshape(lt, s, n_pos, ht, d)
    pv = pred_v.reshape(lt, s, n_pos, ht, d)
    k_hat = jnp.transpose(apply_rope(pk, ang), (0, 1, 3, 2, 4))
    v_hat = jnp.transpose(pv, (0, 1, 3, 2, 4))
    k_nat = jnp.transpose(jnp.take(traces.tgt_k_rope, sample_pos, axis=2), (0, 1, 3, 2, 4))
    v_nat = jnp.transpose(jnp.take(traces.tgt_v, sample_pos, axis=2), (0, 1, 3, 2, 4))

    def one_boundary(c):
        q_c = traces.tgt_q_rope[:, :, c, :, :].reshape(lt, s, ht, q_per_kv, d)
        mask = (sample_pos <= c)[None, None, None, :]
        _, o_nat = attention(q_c, k_nat, v_nat, mask)
        _, o_hat = attention(q_c, k_hat, v_hat, mask)
        return jnp.sum((o_hat - o_nat) ** 2), jnp.sum(o_nat**2)

    num, den = jax.vmap(one_boundary)(boundaries)
    return float(jnp.sqrt(num.sum() / den.sum()))
