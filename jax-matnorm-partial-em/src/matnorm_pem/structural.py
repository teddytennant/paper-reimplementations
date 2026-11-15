"""Section 3.4: submatrix missingness, where separability survives.

When the missing cells of an observation form a rectangle, rows R and columns C,
the missing block of the precision matrix keeps the Kronecker form

    Xi_zz = K_2 (x) K_1 / sigma^2,   K_1 = Xi_1[R, R],  K_2 = Xi_2[C, C],

so the conditional covariance is separable and can be carried as

    Z-hat = sigma^2 (Z-hat_2 (x) Z-hat_1),

with Z-hat_1 and Z-hat_2 updated independently by (8) and (9). Embedding them
back into full size,

    V^R = E_R Z-hat_1 E_R',   V^C = E_C Z-hat_2 E_C',   V-hat = sigma^2 V^C (x) V^R,

the corrections of Section 3.3 collapse to

    R-hat = sigma^2 tr(K_2 Z-hat_2) V^R,   C-hat = sigma^2 tr(K_1 Z-hat_1) V^C.

This is not an approximation. The coordinate updates on a factor converge to the
inverse of that factor's precision, so Z-hat_1 = K_1^{-1} and Z-hat_2 = K_2^{-1}
give sigma^2 (K_2^{-1} (x) K_1^{-1}) = Xi_zz^{-1}, the exact conditional
covariance, and the two corrections then agree with the dense projections to
machine precision. What it buys is size: two factorizations of order |R| and |C|
instead of one of order |R| |C|.

Row and column sets are carried as 0/1 indicator vectors so observations with
different block shapes can be batched. A padded factor inverse,
inv(K * aa' + diag(1 - a)) - diag(1 - a), is exactly E_R Z-hat_1 E_R'.
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import opcount
from .estep import EStepState, _mean_sweep, precisions
from .mstep import _structural_mstep_kernel


def block_mask(arow, acol):
    """The p x q boolean mask of the rectangle R x C."""
    return jnp.outer(jnp.asarray(arow, bool), jnp.asarray(acol, bool))


def _padded_inverse(K, a):
    """Inverse of the principal submatrix K[a, a], embedded back at full size."""
    off = 1.0 - a
    P = K * jnp.outer(a, a) + jnp.diag(off)
    return jnp.linalg.inv(P) - jnp.diag(off)


def _factor_cov_sweep(Z, a, K):
    """One sweep of (8) and (9) on a single Kronecker factor.

    Same two lines as the element-wise version, with Xi_zz replaced by the
    factor precision K[a, a] and every cell of the factor treated as active.
    """
    d = a.shape[0]
    for w in range(d):
        xi = K[:, w] * a
        xi = xi.at[w].set(0.0)
        g = Z @ xi
        row = -g / K[w, w]
        row = row * a
        row = row.at[w].set(1.0 / K[w, w] + (xi @ g) / K[w, w] ** 2)
        row = jnp.where(a[w] > 0, row, jnp.zeros_like(row))
        Z = Z.at[w, :].set(row)
        Z = Z.at[:, w].set(row)
    return Z


def _structural_one(Y, arow, acol, Yhat0, VR0, VC0, params, Xi1, Xi2, mode, mean_sweeps, cov_sweeps):
    M, _, _, sigma2 = params
    mask = jnp.outer(arow, acol) > 0
    D = jnp.where(mask, Yhat0, jnp.where(mask, 0.0, Y)) - M
    D = jax.lax.fori_loop(0, mean_sweeps, lambda _, d: _mean_sweep(d, mask, Xi1, Xi2), D)
    if mode == "block":
        VR = _padded_inverse(Xi1, arow)
        VC = _padded_inverse(Xi2, acol)
    else:
        VR = jax.lax.fori_loop(0, cov_sweeps, lambda _, v: _factor_cov_sweep(v, arow, Xi1), VR0)
        VC = jax.lax.fori_loop(0, cov_sweeps, lambda _, v: _factor_cov_sweep(v, acol, Xi2), VC0)
    return D + M, VR, VC


@partial(jax.jit, static_argnames=("mode", "mean_sweeps", "cov_sweeps"))
def _structural_batch(Y, arow, acol, Yhat0, VR0, VC0, params, mode, mean_sweeps, cov_sweeps):
    Xi1, Xi2 = precisions(params, record=False)
    return jax.vmap(
        _structural_one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
    )(Y, arow, acol, Yhat0, VR0, VC0, params, Xi1, Xi2, mode, mean_sweeps, cov_sweeps)


def init_structural_state(Y, arow, acol, params):
    n, p, q = Y.shape
    mask = jax.vmap(lambda a, b: jnp.outer(a, b) > 0)(arow, acol)
    Yhat = jnp.where(mask, params.M[None], jnp.where(mask, 0.0, Y))
    return (
        Yhat,
        jnp.zeros((n, p, p), dtype=jnp.float64),
        jnp.zeros((n, q, q), dtype=jnp.float64),
    )


def structural_estep(Y, arow, acol, params, state=None, mode="block", mean_sweeps=1, cov_sweeps=1):
    """E-step for submatrix missingness.

    mode="block" takes the two factor inverses directly, which is the exact
    conditional covariance in |R| x |R| and |C| x |C| work. mode="sweep" runs
    `cov_sweeps` passes of (8) and (9) on each factor instead and converges to
    the same thing.

    Returns (Yhat, VR, VC) with V-hat_i = sigma^2 VC_i (x) VR_i.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    arow = jnp.asarray(arow, dtype=jnp.float64)
    acol = jnp.asarray(acol, dtype=jnp.float64)
    if state is None:
        state = init_structural_state(Y, arow, acol, params)
    nr = [int(v) for v in jnp.sum(arow, axis=1)]
    nc = [int(v) for v in jnp.sum(acol, axis=1)]
    if mode == "block":
        for r, c in zip(nr, nc):
            opcount.record("estep", r)
            opcount.record("estep", c)
    precisions(params)
    return _structural_batch(
        Y, arow, acol, state[0], state[1], state[2], params, mode, int(mean_sweeps), int(cov_sweeps)
    )


def structural_corrections(VR, VC, Xi1, Xi2, sigma2):
    """R-hat and C-hat from the separable factors, Section 3.4.

    tr(K_2 Z-hat_2) is written as sum(Xi_2 * V^C) because V^C vanishes outside
    the missing columns, which is the same number with no index gathering.
    """
    R = sigma2 * jnp.sum(Xi2 * VC) * VR
    C = sigma2 * jnp.sum(Xi1 * VR) * VC
    return R, C


def full_V(VR, VC, sigma2):
    """V-hat = sigma^2 V^C (x) V^R, the dense form of the separable covariance."""
    return sigma2 * jnp.kron(VC, VR)


def structural_logdet_precision(arow, acol, Xi1, Xi2, sigma2):
    """log|Xi_z| for a rectangular missing block, without an m x m determinant.

    |K_2 (x) K_1 / sigma^2| = |K_2|^{|R|} |K_1|^{|C|} / sigma^{2 |R| |C|}, so the
    determinant the responsibility update of Section 4.1 needs costs two small
    factorizations instead of one of order |R| |C|.
    """
    nr = jnp.sum(arow)
    nc = jnp.sum(acol)
    ld1 = jnp.linalg.slogdet(Xi1 * jnp.outer(arow, arow) + jnp.diag(1.0 - arow))[1]
    ld2 = jnp.linalg.slogdet(Xi2 * jnp.outer(acol, acol) + jnp.diag(1.0 - acol))[1]
    return nc * ld1 + nr * ld2 - nr * nc * jnp.log(sigma2)


def structural_mstep(state, arow, acol, params, weights=None):
    """M-step (11) and (12) with the separable corrections of Section 3.4."""
    Yhat, VR, VC = state
    n = Yhat.shape[0]
    if weights is None:
        weights = jnp.ones(n, dtype=jnp.float64)
    p, q = params.M.shape
    opcount.record("mstep", p, 2)
    opcount.record("mstep", q)
    return _structural_mstep_kernel(Yhat, VR, VC, jnp.asarray(weights, jnp.float64), params)


def fit_matnorm_structural(Y, arow, acol, params=None, n_iter=50, mode="block",
                           mean_sweeps=1, cov_sweeps=1, track=True):
    """EM for submatrix missingness, the Rect-MPEM of Section 3.4 and Table 1."""
    from .em import init_params
    from .estep import observed_loglik, free_energy

    Y = jnp.asarray(Y, dtype=jnp.float64)
    arow = jnp.asarray(arow, dtype=jnp.float64)
    acol = jnp.asarray(acol, dtype=jnp.float64)
    mask = jax.vmap(lambda a, b: jnp.outer(a, b) > 0)(arow, acol)
    if params is None:
        params = init_params(Y, mask)
    state = init_structural_state(Y, arow, acol, params)
    hist = {"loglik": [], "free_energy": []}
    if track:
        hist["loglik"].append(float(observed_loglik(Y, mask, params)))
    for _ in range(n_iter):
        state = structural_estep(Y, arow, acol, params, state, mode, mean_sweeps, cov_sweeps)
        if track:
            V = jax.vmap(full_V, in_axes=(0, 0, None))(state[1], state[2], params.sigma2)
            hist["free_energy"].append(
                float(free_energy(Y, mask, params, EStepState(state[0], V)))
            )
        params = structural_mstep(state, arow, acol, params)
        if track:
            hist["loglik"].append(float(observed_loglik(Y, mask, params)))
    return params, hist
