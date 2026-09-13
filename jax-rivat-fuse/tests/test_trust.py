"""Trust operators: PSD, PD of H, L+D matvec vs dense."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rivat_fuse.solve import assemble_H
from rivat_fuse.trust import (
    EPS_DEFAULT,
    apply_P,
    apply_P_from_side,
    dense_P,
    dense_P_from_side,
    init_trust_params,
    maps_d_g,
    softplus_pos,
)


D = 8
R_RANK = 3
S_DIM = 3
EPS = 1e-4


def _setup(seed=0):
    key = jax.random.PRNGKey(seed)
    k_p, k_s, k_v = jax.random.split(key, 3)
    params = init_trust_params(k_p, D, R_RANK, S_DIM)
    s = jax.random.normal(k_s, (S_DIM,))
    v = jax.random.normal(k_v, (D,))
    return params, s, v


def test_d_g_positive():
    params, s, _ = _setup()
    d_I, g_I = maps_d_g(params["I"], s)
    d_M, g_M = maps_d_g(params["M"], s)
    assert jnp.all(d_I > 0)
    assert jnp.all(g_I > 0)
    assert jnp.all(d_M > 0)
    assert jnp.all(g_M > 0)


def test_P_I_and_P_M_are_psd():
    params, s, _ = _setup(1)
    P_I = dense_P_from_side(params["I"], s, eps=EPS)
    P_M = dense_P_from_side(params["M"], s, eps=EPS)
    eigs_I = jnp.linalg.eigvalsh(P_I)
    eigs_M = jnp.linalg.eigvalsh(P_M)
    assert float(jnp.min(eigs_I)) >= -1e-5
    assert float(jnp.min(eigs_M)) >= -1e-5


def test_H_is_pd():
    params, s, _ = _setup(2)
    P_I = dense_P_from_side(params["I"], s, eps=EPS)
    P_M = dense_P_from_side(params["M"], s, eps=EPS)
    for lam in (0.0, 1e-2, 1.0):
        H = assemble_H(P_I, P_M, lam)
        min_eig = float(jnp.min(jnp.linalg.eigvalsh(H)))
        assert min_eig > 0


def test_matvec_matches_dense():
    params, s, v = _setup(3)
    d_I, g_I = maps_d_g(params["I"], s)
    U = params["I"]["U"]
    P = dense_P(d_I, g_I, U, eps=EPS)
    mv = apply_P(d_I, g_I, U, v, eps=EPS)
    assert jnp.allclose(mv, P @ v, atol=1e-5, rtol=1e-5)

    side_mv = apply_P_from_side(params["I"], s, v, eps=EPS)
    side_dense = dense_P_from_side(params["I"], s, eps=EPS) @ v
    assert jnp.allclose(side_mv, side_dense, atol=1e-5, rtol=1e-5)


def test_softplus_pos_floor():
    x = jnp.array([-100.0, 0.0, 2.0])
    y = softplus_pos(x, floor=1e-6)
    assert jnp.all(y >= 1e-6)
    assert y.shape == x.shape
