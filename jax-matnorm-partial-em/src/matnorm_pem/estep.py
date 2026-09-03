"""E-steps: the exact one, the partial one of Section 3, and the plug-in.

Three ways to fill in the conditional moments of the missing cells, all sharing
the same interface so the M-step and the likelihood evaluator do not care which
produced them.

exact_estep
    The benchmark. Uses the precision identity (3),
    Sigma_{z|x} = Xi_zz^{-1} and Sigma_zx Sigma_xx^{-1} = -Xi_zz^{-1} Xi_zx,
    so one factorization per observation of order m = number of missing cells
    replaces the o x o factorization in (2). Still pattern-specific, still one
    factorization per distinct pattern per iteration; that is the cost the paper
    is attacking.

partial_estep
    Section 3.1 and 3.2. The conditional mean gets one Gauss-Seidel sweep of (6)
    over the missing cells, the conditional covariance one sweep of (8) and (9).
    Neither touches a factorization: the entries of Xi_zz it needs come straight
    off the Kronecker factors,
        (Xi_zz)_{ab} = (Xi_1)_{i_a i_b} (Xi_2)_{j_a j_b} / sigma^2,
    for missing cells a = (i_a, j_a) and b = (i_b, j_b). Sweeps are warm started
    from the previous iteration, which is what makes one sweep enough.

plugin_estep
    Conditional mean only, covariance correction dropped. This is the "plug-in"
    comparator of Section 3.5 and it is here to show that the small likelihood
    gap of the partial E-step is not what any cheap approximation would give.

Storage note. The paper carries the m x m block Z-hat, whose size changes with
the pattern. To vmap over observations these routines carry V-hat in the fixed
pq x pq layout instead, zero on rows and columns of observed cells, so the
missing block sits inside a padded array. The arithmetic performed is identical;
only the memory layout differs.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import opcount
from .density import vec, unvec


class EStepState(NamedTuple):
    """Conditional moments of a data set under the current parameters.

    Yhat: (n, p, q) completed data, observed cells at their values and missing
        cells at their current conditional mean, the Y-hat_i of (10).
    V: (n, pq, pq) conditional covariance correction V-hat_i, in the padded
        layout described in the module docstring.
    """

    Yhat: jnp.ndarray
    V: jnp.ndarray


def precisions(params, record=True):
    """Xi_1 and Xi_2, the two factor inverses every step needs."""
    _, S1, S2, _ = params
    if record:
        opcount.record("factor", S1.shape[0])
        opcount.record("factor", S2.shape[0])
    return jnp.linalg.inv(S1), jnp.linalg.inv(S2)


def full_precision(params):
    """Xi = (Xi_2 (x) Xi_1) / sigma^2, built only where a dense form is wanted."""
    Xi1, Xi2 = precisions(params, record=False)
    return jnp.kron(Xi2, Xi1) / params.sigma2


def _mask_vec(mask):
    return vec(mask).astype(jnp.float64)


# ---------------------------------------------------------------- exact E-step


def _exact_one(Y, mask, params, Xi):
    """Exact conditional mean and covariance for one observation, via (3).

    The padded system P = mask (x) mask * Xi + diag(1 - mask) is block diagonal
    in the (missing, observed) partition, with blocks Xi_zz and I, so inverting
    it gives Xi_zz^{-1} in the missing block and nothing anywhere else. That is
    Sigma_{z|x} embedded in pq space, the first half of (3). The mean follows
    from the second half:
        vec(Y-hat) = vec(M) + d - V-hat Xi d,
    where d is the centered data with zeros written into the missing cells.
    """
    M = params.M
    p, q = M.shape
    mv = _mask_vec(mask)
    obs = 1.0 - mv
    P = Xi * jnp.outer(mv, mv) + jnp.diag(obs)
    V = jnp.linalg.inv(P) - jnp.diag(obs)
    d = vec(jnp.where(mask, 0.0, jnp.asarray(Y) - M))
    yhat = vec(M) + d - V @ (Xi @ d)
    return unvec(yhat, p, q), V


@jax.jit
def _exact_batch(Y, mask, params):
    Xi = full_precision(params)
    return jax.vmap(_exact_one, in_axes=(0, 0, None, None))(Y, mask, params, Xi)


def exact_estep(Y, mask, params):
    """Exact E-step for every observation. Counts one order-m factorization each."""
    m_per_obs = jnp.sum(mask.reshape(mask.shape[0], -1), axis=1)
    for m in [int(v) for v in m_per_obs]:
        opcount.record("estep", m)
    precisions(params)  # the M-step needs Xi_1 and Xi_2 anyway
    Yhat, V = _exact_batch(jnp.asarray(Y), jnp.asarray(mask, dtype=bool), params)
    return EStepState(Yhat, V)


def exact_estep_by_pattern(Y, mask, params):
    """Exact E-step written the way the paper describes it, pattern by pattern.

    Gathers the actual missing index set of each observation and factorizes the
    real m x m block, no padding. Slower in Python but it is the honest picture
    of what the benchmark costs, and the tests check it against `exact_estep`.
    """
    Y = jnp.asarray(Y)
    mask = jnp.asarray(mask, dtype=bool)
    n, p, q = Y.shape
    M = params.M
    Xi = full_precision(params)
    mu = vec(M)
    Yhat = []
    Vs = []
    for i in range(n):
        mv = vec(mask[i])
        idx = jnp.nonzero(mv, size=int(mv.sum()))[0]
        m = int(idx.shape[0])
        d = vec(jnp.where(mask[i], 0.0, Y[i] - M))
        V = jnp.zeros((p * q, p * q), dtype=jnp.float64)
        yhat = mu + d
        if m > 0:
            opcount.record("estep", m)
            Xzz = Xi[jnp.ix_(idx, idx)]
            Zi = jnp.linalg.inv(Xzz)
            V = V.at[jnp.ix_(idx, idx)].set(Zi)
            yhat = yhat.at[idx].add(-Zi @ (Xi[idx, :] @ d))
        Yhat.append(unvec(yhat, p, q))
        Vs.append(V)
    return EStepState(jnp.stack(Yhat), jnp.stack(Vs))


# -------------------------------------------------------------- partial E-step


def _mean_sweep(D, mask, Xi1, Xi2):
    """One Gauss-Seidel sweep of (6) over the missing cells.

    G = Xi_1 (Y-hat - M) Xi_2 is the gradient of (5), the second derivative in
    cell (i, j) is (Xi_1)_ii (Xi_2)_jj, so the exact minimizer along that
    coordinate is a step of eta_ij = -G_ij / [(Xi_1)_ii (Xi_2)_jj]. Only G_ij is
    needed, one row-vector times matrix times column-vector, so the sweep never
    forms G. Cells are visited in column-stacking order and each uses the values
    the earlier cells just took.
    """
    p, q = D.shape
    for j in range(q):
        for i in range(p):
            g = Xi1[i] @ D @ Xi2[:, j]
            eta = -g / (Xi1[i, i] * Xi2[j, j])
            D = D.at[i, j].add(jnp.where(mask[i, j], eta, 0.0))
    return D


def _cov_sweep(V, mask, Xi1, Xi2, sigma2):
    """One sweep of the covariance updates (8) and (9) over the missing cells.

    For active cell w, xi_w is the column of Xi_zz at w with the w entry
    dropped, which is (Xi_1[:, i] Xi_2[j, :]') / sigma^2 restricted to the other
    missing cells, and W is the current covariance of those cells. The updates
        w-hat_vec = -W xi_w / xi_ww                                        (8)
        w-hat     = 1 / xi_ww + xi_w' W xi_w / xi_ww^2                     (9)
    are written here on the padded pq x pq array: rows and columns of observed
    cells are zero, so W xi_w and xi_w' W xi_w pick out exactly the missing
    block, and zeroing xi_w at w drops row and column w as the derivation
    requires. No inverse of W appears, matching the remark after (9).
    """
    p, q = mask.shape
    mv = _mask_vec(mask)
    for j in range(q):
        for i in range(p):
            w = j * p + i
            xi = jnp.outer(Xi1[:, i], Xi2[:, j]).T.reshape(-1) / sigma2
            xi = xi * mv
            xi = xi.at[w].set(0.0)
            xi_ww = Xi1[i, i] * Xi2[j, j] / sigma2
            g = V @ xi
            row = -g / xi_ww
            diag = 1.0 / xi_ww + (xi @ g) / xi_ww**2
            row = row * mv
            row = row.at[w].set(diag)
            row = jnp.where(mask[i, j], row, jnp.zeros_like(row))
            V = V.at[w, :].set(row)
            V = V.at[:, w].set(row)
    return V


def _partial_one(Y, mask, state_Yhat, state_V, params, Xi1, Xi2, mean_sweeps, cov_sweeps):
    M, _, _, sigma2 = params
    D = jnp.where(mask, state_Yhat, jnp.where(mask, 0.0, Y)) - M
    # fori_loop rather than a Python loop so that asking for many sweeps costs
    # iterations, not compile time
    D = jax.lax.fori_loop(0, mean_sweeps, lambda _, d: _mean_sweep(d, mask, Xi1, Xi2), D)
    V = jax.lax.fori_loop(
        0, cov_sweeps, lambda _, v: _cov_sweep(v, mask, Xi1, Xi2, sigma2), state_V
    )
    return D + M, V


@partial(jax.jit, static_argnames=("mean_sweeps", "cov_sweeps"))
def _partial_batch(Y, mask, state, params, mean_sweeps, cov_sweeps):
    Xi1, Xi2 = precisions(params, record=False)
    return jax.vmap(_partial_one, in_axes=(0, 0, 0, 0, None, None, None, None, None))(
        Y, mask, state.Yhat, state.V, params, Xi1, Xi2, mean_sweeps, cov_sweeps
    )


def partial_estep(Y, mask, params, state, mean_sweeps=1, cov_sweeps=1):
    """Partial E-step of Section 3, warm started from `state`.

    No pattern-specific factorization is recorded because none happens; the only
    inverses are Xi_1 and Xi_2, which the M-step needs regardless.
    """
    precisions(params)
    Yhat, V = _partial_batch(
        jnp.asarray(Y),
        jnp.asarray(mask, dtype=bool),
        state,
        params,
        int(mean_sweeps),
        int(cov_sweeps),
    )
    return EStepState(Yhat, V)


def plugin_estep(Y, mask, params, state, mean_sweeps=1):
    """Conditional mean only, V-hat left at zero (the plug-in of Section 3.5)."""
    st = partial_estep(Y, mask, params, state, mean_sweeps=mean_sweeps, cov_sweeps=0)
    return EStepState(st.Yhat, jnp.zeros_like(st.V))


def init_estep_state(Y, mask, params):
    """Start the missing cells at the current mean and their covariance at zero.

    The first covariance sweep turns the zero matrix into the diagonal
    1 / xi_ww and then fills in cross terms, so no special initialization is
    needed for (8) and (9).
    """
    Y = jnp.asarray(Y)
    mask = jnp.asarray(mask, dtype=bool)
    n, p, q = Y.shape
    Yhat = jnp.where(mask, params.M[None], jnp.where(mask, 0.0, Y))
    V = jnp.zeros((n, p * q, p * q), dtype=jnp.float64)
    return EStepState(Yhat, V)


# ------------------------------------------------------- likelihood evaluators


def _observed_loglik_one(Y, mask, params, Xi):
    """Log density of the observed cells of one matrix, exactly.

    Section 4.1 evaluates this without touching the o x o observed block, using
    log|Sigma_x| = log|Sigma| + log|Xi_z| and the fact that the completed
    quadratic form at the conditional mean equals the observed quadratic form:
        log phi(y_x) = -[o log 2pi + log|Sigma| + log|Xi_z| + Q] / 2,
        Q = tr(Xi_1 (Y-hat - M) Xi_2 (Y-hat - M)') / sigma^2.
    Y-hat here is always the exact conditional mean, whichever E-step produced
    the parameters, so this is a fair scoreboard.
    """
    M, S1, S2, sigma2 = params
    p, q = M.shape
    mv = _mask_vec(mask)
    obs = 1.0 - mv
    P = Xi * jnp.outer(mv, mv) + jnp.diag(obs)
    sign, logdet_Xz = jnp.linalg.slogdet(P)
    V = jnp.linalg.inv(P) - jnp.diag(obs)
    d = vec(jnp.where(mask, 0.0, Y - M))
    dhat = d - V @ (Xi @ d)
    quad = dhat @ (Xi @ dhat)
    o = jnp.sum(obs)
    logdet_S = p * q * jnp.log(sigma2) + q * jnp.linalg.slogdet(S1)[1] + p * jnp.linalg.slogdet(S2)[1]
    return -0.5 * (o * jnp.log(2.0 * jnp.pi) + logdet_S + logdet_Xz + quad)


@jax.jit
def _observed_loglik_batch(Y, mask, params):
    Xi = full_precision(params)
    return jax.vmap(_observed_loglik_one, in_axes=(0, 0, None, None))(Y, mask, params, Xi)


def observed_loglik(Y, mask, params, per_observation=False):
    """Observed-data log-likelihood, the quantity EM must not decrease."""
    Y = jnp.asarray(Y)
    mask = jnp.asarray(mask, dtype=bool)
    m_per_obs = jnp.sum(mask.reshape(mask.shape[0], -1), axis=1)
    for m in [int(v) for v in m_per_obs]:
        opcount.record("eval", m)
    ll = _observed_loglik_batch(jnp.where(mask, 0.0, Y), mask, params)
    return ll if per_observation else jnp.sum(ll)


def _free_energy_one(Y, mask, state_Yhat, state_V, params, Xi):
    M, S1, S2, sigma2 = params
    p, q = M.shape
    mv = _mask_vec(mask)
    obs = 1.0 - mv
    d = vec(jnp.where(mask, state_Yhat - M, jnp.where(mask, 0.0, Y) - M))
    quad = d @ (Xi @ d)
    trace = jnp.sum(Xi * state_V)
    m = jnp.sum(mv)
    logdet_Z = jnp.linalg.slogdet(state_V + jnp.diag(obs))[1]
    logdet_S = p * q * jnp.log(sigma2) + q * jnp.linalg.slogdet(S1)[1] + p * jnp.linalg.slogdet(S2)[1]
    complete = -0.5 * (p * q * jnp.log(2.0 * jnp.pi) + logdet_S + quad + trace)
    entropy = 0.5 * (m * jnp.log(2.0 * jnp.pi * jnp.e) + logdet_Z)
    return complete + entropy


@jax.jit
def _free_energy_batch(Y, mask, state, params):
    Xi = full_precision(params)
    return jax.vmap(_free_energy_one, in_axes=(0, 0, 0, 0, None, None))(
        Y, mask, state.Yhat, state.V, params, Xi
    )


def free_energy(Y, mask, params, state):
    """The variational bound the partial E-step actually optimizes.

    F = E_q[log p(y_x, z | theta)] + H(q) for q = N(z-hat, Z-hat), which is the
    observed log-likelihood minus the KL divergence (4). Every partial update in
    Section 3.1 and 3.2 lowers (4) and the M-step raises the first term, so F is
    non-decreasing across a partial EM iteration even where the observed
    log-likelihood has no such guarantee. `free_energy <= observed_loglik`
    always, with equality when the sweeps have converged.
    """
    Y = jnp.asarray(Y)
    mask = jnp.asarray(mask, dtype=bool)
    return jnp.sum(_free_energy_batch(jnp.where(mask, 0.0, Y), mask, state, params))
