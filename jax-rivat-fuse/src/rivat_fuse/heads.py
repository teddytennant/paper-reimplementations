"""Stabilized quadratic task heads (eq 16) and optional task-coupling (eq 15).

f = w^T h + h^T S h + b
S = Diag(s) + L L^T  (not a free dense d x d matrix)
Ω_task = tr(W L_T W^T)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _normal(key, shape, scale=0.1):
    return scale * jax.random.normal(key, shape, dtype=jnp.float32)


def init_quadratic_head(key, d, rank=4, n_classes=1):
    kw, kL, kd = jax.random.split(key, 3)
    return {
        "w": _normal(kw, (n_classes, d), scale=0.1),
        "L": _normal(kL, (n_classes, d, rank), scale=0.05),
        "s_diag": _normal(kd, (n_classes, d), scale=0.05),
        "b": jnp.zeros((n_classes,), dtype=jnp.float32),
    }


def quadratic_form(h, s_diag, L):
    """h^T (Diag(s_diag) + L L^T) h for one class. h (d,), L (d, r)."""
    diag_term = jnp.dot(h * s_diag, h)
    low_rank = jnp.sum((L.T @ h) ** 2)
    return diag_term + low_rank


def quadratic_head(params, h):
    """Scores (n_classes,). S=0 (s_diag=0, L=0) reduces to linear w h + b."""
    linear = params["w"] @ h
    # vmap over classes: L (C, d, r), s_diag (C, d)
    quad = jax.vmap(lambda sd, L: quadratic_form(h, sd, L))(
        params["s_diag"], params["L"]
    )
    return linear + quad + params["b"]


def task_coupling(W, L_T):
    """Ω_task = tr(W L_T W^T). W is (d, T) or (C, d) task-coefficient matrix."""
    return jnp.trace(W @ L_T @ W.T)
