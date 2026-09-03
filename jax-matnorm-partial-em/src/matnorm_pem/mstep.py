"""M-step: (10), (11), (12) and the scale scalar.

The conditional covariance V-hat_i lives in pq x pq space and has to be pushed
back onto the p x p and q x q factors. Section 3.3 does that by projection:

    R-hat_{i,kl} = tr(Xi_2 V-hat^R_{i,kl}),   V-hat^R_{i,kl} the q x q slice with
                                              entries Cov(Y_{k a}, Y_{l b})
    C-hat_{i,kl} = tr(Xi_1 V-hat^C_{i,kl}),   V-hat^C_{i,kl} the p x p block
                                              Cov(Y_{. k}, Y_{. l})

which are the element-wise derivative projections written out in Section 2.2.
Under column stacking, viewing V-hat as a (q, p, q, p) array indexed [a, k, b, l]
turns both projections into one einsum.

Order matters: (11) uses the current Xi_2, then (12) uses the Xi_1 that (11)
just produced, so the pair is a conditional maximization and the expected
complete-data log-likelihood cannot go down. The determinant normalizations
impose |S1| = |S2| = 1 from Section 2.4 and leave the scale in
sigma^2 = |Sigma-tilde_2|^{1/q}.

Sanity anchor: for a fully missing observation V-hat = sigma^2 (S2 (x) S1), so
R-hat = sigma^2 q S1 and C-hat = sigma^2 p S2 and (11) and (12) hand back
sigma^2 S1 and sigma^2 S2.
"""

import jax
import jax.numpy as jnp

from . import opcount
from .params import MatNormParams


def _as4(V, p, q):
    return V.reshape(q, p, q, p)


def row_correction(V, Xi2, p, q):
    """R-hat of (11): project V-hat onto the row space with the column precision."""
    return jnp.einsum("ab,akbl->kl", Xi2, _as4(V, p, q))


def col_correction(V, Xi1, p, q):
    """C-hat of (12): project V-hat onto the column space with the row precision."""
    return jnp.einsum("ab,kalb->kl", Xi1, _as4(V, p, q))


def _weighted_mean(weights, A):
    return jnp.einsum("n,n...->...", weights, A) / jnp.sum(weights)


def _update_S1(D, weights, R, Xi2, p, q):
    """(11) up to the normalization, given the row-space corrections R-hat_i."""
    S1t = _weighted_mean(weights, jnp.einsum("npa,ab,nqb->npq", D, Xi2, D) + R) / q
    S1t = 0.5 * (S1t + S1t.T)
    return S1t / jnp.linalg.det(S1t) ** (1.0 / p)


def _update_S2(D, weights, C, Xi1, p, q):
    """(12) plus sigma^2, given the column-space corrections C-hat_i."""
    S2t = _weighted_mean(weights, jnp.einsum("nap,ab,nbq->npq", D, Xi1, D) + C) / p
    S2t = 0.5 * (S2t + S2t.T)
    det2 = jnp.linalg.det(S2t) ** (1.0 / q)
    return S2t / det2, det2


@jax.jit
def _mstep_kernel(Yhat, V, weights, params):
    p, q = params.M.shape
    Xi1 = jnp.linalg.inv(params.S1)
    Xi2 = jnp.linalg.inv(params.S2)

    M = _weighted_mean(weights, Yhat)  # (10)
    D = Yhat - M

    R = jax.vmap(row_correction, in_axes=(0, None, None, None))(V, Xi2, p, q)
    S1 = _update_S1(D, weights, R, Xi2, p, q)

    Xi1n = jnp.linalg.inv(S1)
    C = jax.vmap(col_correction, in_axes=(0, None, None, None))(V, Xi1n, p, q)
    S2, sigma2 = _update_S2(D, weights, C, Xi1n, p, q)
    return MatNormParams(M, S1, S2, sigma2)


@jax.jit
def _structural_mstep_kernel(Yhat, VR, VC, weights, params):
    """Same M-step, with the separable corrections of Section 3.4.

    R-hat_i = sigma^2 tr(K_2 Z-hat_2) V^R_i and C-hat_i = sigma^2 tr(K_1 Z-hat_1)
    V^C_i. The traces need no index sets: V^R and V^C are zero outside the
    missing rows and columns, so tr(K_2 Z-hat_2) = sum(Xi_2 * V^C) already
    restricts to C, and likewise for the row factor.
    """
    p, q = params.M.shape
    Xi1 = jnp.linalg.inv(params.S1)
    Xi2 = jnp.linalg.inv(params.S2)
    sigma2 = params.sigma2

    M = _weighted_mean(weights, Yhat)
    D = Yhat - M

    R = sigma2 * jnp.einsum("n,npq->npq", jnp.einsum("ab,nab->n", Xi2, VC), VR)
    S1 = _update_S1(D, weights, R, Xi2, p, q)

    Xi1n = jnp.linalg.inv(S1)
    C = sigma2 * jnp.einsum("n,npq->npq", jnp.einsum("ab,nab->n", Xi1n, VR), VC)
    S2, sigma2n = _update_S2(D, weights, C, Xi1n, p, q)
    return MatNormParams(M, S1, S2, sigma2n)


def mstep(state, params, weights=None):
    """One M-step from a dense E-step state. `weights` are the u-hat_ig of (14)."""
    n = state.Yhat.shape[0]
    if weights is None:
        weights = jnp.ones(n, dtype=jnp.float64)
    p, q = params.M.shape
    opcount.record("mstep", p, 2)  # Xi_1 before and after the S1 update
    opcount.record("mstep", q)
    return _mstep_kernel(state.Yhat, state.V, jnp.asarray(weights, jnp.float64), params)
