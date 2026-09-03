"""Section 3.4: the submatrix specialization is an identity, not an approximation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matnorm_pem as mp
from matnorm_pem import synth
from matnorm_pem.estep import full_precision
from matnorm_pem.mstep import row_correction, col_correction
from matnorm_pem.structural import (
    block_mask,
    full_V,
    structural_corrections,
    structural_estep,
    structural_logdet_precision,
    fit_matnorm_structural,
)


def _rect_data(p, q, nr, nc, n=8, seed=0):
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(seed), params, n)
    arow, acol = synth.rect_indicators(jr.key(seed + 300), n, p, q, nr, nc)
    mask = jax.vmap(lambda a, b: block_mask(a, b))(arow, acol)
    return params, Y, arow, acol, mask


SHAPES = [(4, 5, 2, 3), (5, 4, 3, 2), (4, 4, 2, 2), (5, 5, 3, 4)]


@pytest.mark.parametrize("p,q,nr,nc", SHAPES)
def test_separable_covariance_equals_the_dense_conditional(p, q, nr, nc):
    """sigma^2 (Z-hat_2 (x) Z-hat_1) is the exact conditional covariance.

    Xi_zz = K_2 (x) K_1 / sigma^2 for a rectangular block, so inverting the two
    factors of order |R| and |C| separately has to reproduce, to machine
    precision, the single order |R||C| inverse the general E-step takes. This is
    an algebraic identity, hence the tight tolerance.
    """
    params, Y, arow, acol, mask = _rect_data(p, q, nr, nc)
    dense = mp.exact_estep(Y, mask, params)
    _, VR, VC = structural_estep(Y, arow, acol, params, mode="block", mean_sweeps=0)
    V = jax.vmap(full_V, in_axes=(0, 0, None))(VR, VC, params.sigma2)
    assert float(jnp.max(jnp.abs(V - dense.V))) < 1e-11


@pytest.mark.parametrize("p,q,nr,nc", SHAPES)
def test_separable_corrections_equal_the_dense_projections(p, q, nr, nc):
    """R-hat = sigma^2 tr(K_2 Z-hat_2) V^R and its column twin, against (11) and (12).

    The dense side runs the full (q, p, q, p) projections of Section 3.3 on the
    exact V-hat; the separable side never builds a pq x pq matrix.
    """
    params, Y, arow, acol, mask = _rect_data(p, q, nr, nc, seed=5)
    dense = mp.exact_estep(Y, mask, params)
    _, VR, VC = structural_estep(Y, arow, acol, params, mode="block", mean_sweeps=0)
    Xi1 = jnp.linalg.inv(params.S1)
    Xi2 = jnp.linalg.inv(params.S2)
    R_dense = jax.vmap(row_correction, in_axes=(0, None, None, None))(dense.V, Xi2, p, q)
    C_dense = jax.vmap(col_correction, in_axes=(0, None, None, None))(dense.V, Xi1, p, q)
    R_sep, C_sep = jax.vmap(
        lambda vr, vc: structural_corrections(vr, vc, Xi1, Xi2, params.sigma2)
    )(VR, VC)
    assert float(jnp.max(jnp.abs(R_dense - R_sep))) < 1e-11
    assert float(jnp.max(jnp.abs(C_dense - C_sep))) < 1e-11


@pytest.mark.parametrize("p,q,nr,nc", SHAPES)
def test_separable_logdet_matches_the_dense_one(p, q, nr, nc):
    """log|Xi_z| from the two factors, the determinant the mixture E-step needs."""
    params, Y, arow, acol, mask = _rect_data(p, q, nr, nc, seed=9)
    Xi = full_precision(params)
    Xi1 = jnp.linalg.inv(params.S1)
    Xi2 = jnp.linalg.inv(params.S2)
    for i in range(Y.shape[0]):
        mv = mp.vec(mask[i]).astype(jnp.float64)
        dense = float(jnp.linalg.slogdet(Xi * jnp.outer(mv, mv) + jnp.diag(1.0 - mv))[1])
        sep = float(structural_logdet_precision(arow[i], acol[i], Xi1, Xi2, params.sigma2))
        assert abs(dense - sep) < 1e-10 * max(1.0, abs(dense))


def test_factor_sweeps_converge_to_the_factor_inverses():
    """Running (8) and (9) on each factor reaches the same place as inverting it."""
    params, Y, arow, acol, _ = _rect_data(5, 4, 3, 2, seed=13)
    _, VR_block, VC_block = structural_estep(Y, arow, acol, params, mode="block", mean_sweeps=0)
    state = None
    for _ in range(30):
        state = structural_estep(Y, arow, acol, params, state, mode="sweep",
                                 mean_sweeps=1, cov_sweeps=1)
    assert float(jnp.max(jnp.abs(state[1] - VR_block))) < 1e-10
    assert float(jnp.max(jnp.abs(state[2] - VC_block))) < 1e-10


def test_structural_mean_sweeps_reach_the_exact_conditional_mean():
    """The Gauss-Seidel sweep of (6) does not care about the pattern shape."""
    params, Y, arow, acol, mask = _rect_data(4, 5, 2, 3, seed=17)
    dense = mp.exact_estep(Y, mask, params)
    Yhat, _, _ = structural_estep(Y, arow, acol, params, mode="block", mean_sweeps=200)
    assert float(jnp.max(jnp.abs(Yhat - dense.Yhat))) < 1e-9


@pytest.mark.parametrize("seed", [0, 1])
def test_structural_em_is_monotone_and_matches_exact_em(seed):
    """Rect-MPEM against the general E-step on the same submatrix-missing data."""
    p, q, n = 4, 5, 120
    params, Y, arow, acol, mask = _rect_data(p, q, 2, 3, n=n, seed=seed)
    est, hist = fit_matnorm_structural(Y, arow, acol, n_iter=40, mode="block")
    scale = max(1.0, abs(hist["loglik"][-1]))
    assert float(np.min(np.diff(np.asarray(hist["loglik"])))) > -1e-9 * scale
    assert float(np.min(np.diff(np.asarray(hist["free_energy"])))) > -1e-9 * scale
    _, h_exact = mp.fit_matnorm(Y, mask, method="exact", n_iter=40)
    assert abs(hist["loglik"][-1] - h_exact["loglik"][-1]) / n < 1e-3
