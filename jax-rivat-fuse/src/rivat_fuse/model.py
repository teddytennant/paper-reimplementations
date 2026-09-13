"""RiVaT-Fuse kernel: init a pytree and fuse (z_I, z_M) into h*."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rivat_fuse.heads import init_quadratic_head, quadratic_head
from rivat_fuse.interaction import init_interaction_params, interaction
from rivat_fuse.solve import assemble_H, assemble_b, solve_latent
from rivat_fuse.trust import (
    EPS_DEFAULT,
    apply_P_from_side,
    dense_P_from_side,
    init_trust_params,
)


def init_params(
    key,
    d,
    s_dim=3,
    r_rank=4,
    K=2,
    r_k=4,
    head_rank=4,
    n_classes=1,
    eps=EPS_DEFAULT,
    lam=1e-2,
):
    kt, ki, kh = jax.random.split(key, 3)
    return {
        "trust": init_trust_params(kt, d, r_rank, s_dim),
        "interaction": init_interaction_params(ki, d, s_dim, K=K, r_k=r_k),
        "head": init_quadratic_head(kh, d, rank=head_rank, n_classes=n_classes),
        "eps": jnp.asarray(eps, dtype=jnp.float32),
        "lam": jnp.asarray(lam, dtype=jnp.float32),
    }


def _s_dim(params):
    return params["trust"]["I"]["d_W"].shape[1]


def _default_s(params, z_I):
    return jnp.ones((_s_dim(params),), dtype=z_I.dtype) * 0.5


def fuse_one(params, z_I, z_M, tokens=None, s=None):
    """Single-sample latent solve. z_I, z_M: (d,), tokens: (L, d) or None."""
    if s is None:
        s = _default_s(params, z_I)
    eps = params["eps"]
    lam = params["lam"]
    P_I = dense_P_from_side(params["trust"]["I"], s, eps=eps)
    P_M = dense_P_from_side(params["trust"]["M"], s, eps=eps)
    t_vec = interaction(params["interaction"], z_I, z_M, tokens=tokens, s=s)
    H = assemble_H(P_I, P_M, lam)
    b = assemble_b(P_I, P_M, z_I, z_M, t_vec)
    return solve_latent(H, b)


def fuse_one_from_matvec(params, z_I, z_M, tokens=None, s=None):
    """Same solve, but b uses P matvecs (still forms H densely for the solve)."""
    if s is None:
        s = _default_s(params, z_I)
    eps = params["eps"]
    lam = params["lam"]
    P_I = dense_P_from_side(params["trust"]["I"], s, eps=eps)
    P_M = dense_P_from_side(params["trust"]["M"], s, eps=eps)
    t_vec = interaction(params["interaction"], z_I, z_M, tokens=tokens, s=s)
    H = assemble_H(P_I, P_M, lam)
    b = (
        apply_P_from_side(params["trust"]["I"], s, z_I, eps=eps)
        + apply_P_from_side(params["trust"]["M"], s, z_M, eps=eps)
        + t_vec
    )
    return solve_latent(H, b)


def fuse(params, z_I, z_M, tokens=None, s=None):
    """Fuse one sample (d,) or a batch (B, d). tokens (L, d) or (B, L, d)."""
    if z_I.ndim == 1:
        return fuse_one(params, z_I, z_M, tokens=tokens, s=s)

    if s is None:
        s = jnp.ones((z_I.shape[0], _s_dim(params)), dtype=z_I.dtype) * 0.5

    if tokens is None:

        def _one_no_tok(z_i, z_m, ss):
            return fuse_one(params, z_i, z_m, tokens=None, s=ss)

        return jax.vmap(_one_no_tok)(z_I, z_M, s)

    def _one(z_i, z_m, tok, ss):
        return fuse_one(params, z_i, z_m, tokens=tok, s=ss)

    return jax.vmap(_one)(z_I, z_M, tokens, s)


def predict(params, z_I, z_M, tokens=None, s=None):
    """Fuse then apply the quadratic head. Returns (n_classes,) or (B, n_classes)."""
    h = fuse(params, z_I, z_M, tokens=tokens, s=s)
    if h.ndim == 1:
        return quadratic_head(params["head"], h)
    return jax.vmap(lambda hh: quadratic_head(params["head"], hh))(h)
