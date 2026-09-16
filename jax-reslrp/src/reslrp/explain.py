"""Attribute a residual-MLP logit back to the input with ResLRP."""

from __future__ import annotations

import jax.numpy as jnp

from reslrp.linear import lrp_eps
from reslrp.models import residual_mlp
from reslrp.residual import lrp_gamma


def attribute(params, x, class_idx, gamma, eps: float = 1e-9):
    """Input relevance for ``class_idx``, conservative through residuals.

    Seeds relevance with the chosen logit (others zero) and walks backward:
    dense LRP-ε on the head and on each residual MLP, identity through GELU,
    LRP-γ (Eq. 3) at each skip add. Returns relevance with the same shape as
    ``x``.
    """
    logits, caches = residual_mlp(params, x, cache=True)
    r_logits = jnp.zeros_like(logits)
    logit_c = logits[..., class_idx]
    r_logits = r_logits.at[..., class_idx].set(logit_c)

    h = caches[-1]["z_out"] if caches else x
    r_h = lrp_eps(h, params["W_head"], None, r_logits, eps=eps)

    for block, c in zip(reversed(params["blocks"]), reversed(caches)):
        r_in, r_up = lrp_gamma(c["z_in"], c["z_up"], r_h, gamma, eps=eps)
        r_hid = lrp_eps(c["hid"], block["W2"], None, r_up, eps=eps)
        r_from_up = lrp_eps(c["z_in"], block["W1"], None, r_hid, eps=eps)
        r_h = r_in + r_from_up
    return r_h
