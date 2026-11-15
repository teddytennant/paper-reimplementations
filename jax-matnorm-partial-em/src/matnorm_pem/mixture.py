"""Section 4: mixtures of matrix normals with missing cells.

The density is (13), a G-component mixture of the constrained matrix normal of
Section 4.1. Membership and missing cells are both latent, so one E-step does
two things: fill in each observation's missing cells inside every component,
and score the observation against every component.

The scoring uses the observed-data density, not the density at the imputed
matrix, which is the point of the determinant identity

    log|Sigma^i_{g,x}| = log|Sigma_g| + log|Xi^i_{g,z}|,

giving

    log phi(y_{ig,x}) = -[o log 2pi + log|Sigma_g| + log|Xi_{ig,z}| + Q_ig] / 2,
    Q_ig = tr(Xi_{1g} (Y-hat_ig - M_g) Xi_{2g} (Y-hat_ig - M_g)') / sigma_g^2,

and then u-hat_ig = pi_g phi_ig / sum_h pi_h phi_ih. With the exact E-step
Y-hat_ig is the exact conditional mean and Q_ig is the exact observed quadratic
form; with the partial E-step it is whatever the sweeps have reached, so the
responsibilities inherit the same small approximation as everything else.

The M-step is (14), (15), (16) plus pi_g = N_g / N, which is the single
component M-step weighted by u-hat_ig.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from . import opcount
from .estep import (
    EStepState,
    exact_estep,
    partial_estep,
    init_estep_state,
    full_precision,
    _mask_vec,
)
from .mstep import mstep
from .params import MatNormParams
from .density import vec


class MixtureParams(NamedTuple):
    """Mixture of (13): weights and one MatNormParams with a leading G axis."""

    pi: jnp.ndarray
    comps: MatNormParams

    @property
    def G(self) -> int:
        return self.pi.shape[0]


def _component(params: MixtureParams, g: int) -> MatNormParams:
    return MatNormParams(*[leaf[g] for leaf in params.comps])


def _component_loglik_one(Y, mask, Yhat, params, Xi):
    """log phi(y_x | theta_g) evaluated at a given completion, Section 4.1."""
    M, S1, S2, sigma2 = params
    p, q = M.shape
    mv = _mask_vec(mask)
    obs = 1.0 - mv
    logdet_Xz = jnp.linalg.slogdet(Xi * jnp.outer(mv, mv) + jnp.diag(obs))[1]
    d = vec(jnp.where(mask, Yhat, jnp.where(mask, 0.0, Y)) - M)
    quad = d @ (Xi @ d)
    logdet_S = (
        p * q * jnp.log(sigma2)
        + q * jnp.linalg.slogdet(S1)[1]
        + p * jnp.linalg.slogdet(S2)[1]
    )
    return -0.5 * (jnp.sum(obs) * jnp.log(2.0 * jnp.pi) + logdet_S + logdet_Xz + quad)


@jax.jit
def _component_loglik(Y, mask, Yhat, params):
    Xi = full_precision(params)
    return jax.vmap(_component_loglik_one, in_axes=(0, 0, 0, None, None))(
        Y, mask, Yhat, params, Xi
    )


def _log_sum_exp(a, axis):
    mx = jnp.max(a, axis=axis, keepdims=True)
    return jnp.squeeze(mx, axis) + jnp.log(jnp.sum(jnp.exp(a - mx), axis=axis))


def mixture_loglik(Y, mask, params: MixtureParams):
    """Observed-data log-likelihood of (13), scored with exact completions."""
    from .estep import observed_loglik

    lls = jnp.stack(
        [observed_loglik(Y, mask, _component(params, g), per_observation=True)
         for g in range(params.G)],
        axis=1,
    )
    return float(jnp.sum(_log_sum_exp(jnp.log(params.pi)[None] + lls, axis=1)))


def kmeans_responsibilities(Y, mask, G, key, n_iter=15, smoothing=0.05):
    """Hard k-means on the mean-filled vectorized data, softened a little.

    A starting point, not part of the model. Missing cells go to the column
    mean of the observed values before clustering.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    n, p, q = Y.shape
    obs = (~mask).astype(jnp.float64)
    colmean = jnp.sum(jnp.where(mask, 0.0, Y), axis=0) / jnp.maximum(jnp.sum(obs, axis=0), 1.0)
    X = jnp.where(mask, colmean[None], Y).reshape(n, p * q)
    idx = jr.choice(key, n, (G,), replace=False)
    C = X[idx]
    for _ in range(n_iter):
        d = jnp.sum((X[:, None, :] - C[None]) ** 2, axis=-1)
        lab = jnp.argmin(d, axis=1)
        onehot = jax.nn.one_hot(lab, G, dtype=jnp.float64)
        counts = jnp.maximum(jnp.sum(onehot, axis=0), 1.0)
        C = (onehot.T @ X) / counts[:, None]
    d = jnp.sum((X[:, None, :] - C[None]) ** 2, axis=-1)
    onehot = jax.nn.one_hot(jnp.argmin(d, axis=1), G, dtype=jnp.float64)
    return (1.0 - smoothing) * onehot + smoothing / G


def init_mixture(Y, mask, resp):
    """One weighted M-step from starting responsibilities gives the first fit."""
    from .em import init_params

    Y = jnp.asarray(Y, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    G = resp.shape[1]
    base = init_params(Y, mask)
    comps = []
    for g in range(G):
        st = init_estep_state(Y, mask, base)
        comps.append(mstep(st, base, weights=resp[:, g]))
    pi = jnp.sum(resp, axis=0) / resp.shape[0]
    stacked = MatNormParams(*[jnp.stack([getattr(c, f) for c in comps]) for f in MatNormParams._fields])
    return MixtureParams(pi, stacked)


def mixture_estep(Y, mask, params: MixtureParams, states, method="partial",
                  mean_sweeps=1, cov_sweeps=1):
    """Component-wise conditional moments and the responsibilities of Section 4.1."""
    G = params.G
    new_states = []
    loglls = []
    for g in range(G):
        pg = _component(params, g)
        if method == "exact":
            st = exact_estep(Y, mask, pg)
        else:
            st = partial_estep(Y, mask, pg, states[g], mean_sweeps, cov_sweeps)
        new_states.append(st)
        m_per_obs = jnp.sum(mask.reshape(mask.shape[0], -1), axis=1)
        for m in [int(v) for v in m_per_obs]:
            opcount.record("resp", m)
        loglls.append(_component_loglik(jnp.where(mask, 0.0, Y), mask, st.Yhat, pg))
    L = jnp.stack(loglls, axis=1) + jnp.log(params.pi)[None]
    resp = jnp.exp(L - _log_sum_exp(L, axis=1)[:, None])
    return new_states, resp


def mixture_mstep(states, params: MixtureParams, resp):
    """(14), (15), (16) and pi_g = N_g / N."""
    G = params.G
    comps = [mstep(states[g], _component(params, g), weights=resp[:, g]) for g in range(G)]
    pi = jnp.sum(resp, axis=0) / resp.shape[0]
    stacked = MatNormParams(*[jnp.stack([getattr(c, f) for c in comps]) for f in MatNormParams._fields])
    return MixtureParams(pi, stacked)


def fit_mixture(Y, mask, G, key, method="partial", n_iter=40, mean_sweeps=1,
                cov_sweeps=1, track=True):
    """Fit (13) with missing cells, clustering and imputing at the same time.

    Returns (params, resp, history). history["loglik"] is the mixture
    observed-data log-likelihood after each M-step, always evaluated exactly.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    resp = kmeans_responsibilities(Y, mask, G, key)
    params = init_mixture(Y, mask, resp)
    states = [init_estep_state(Y, mask, _component(params, g)) for g in range(G)]
    hist = {"loglik": []}
    if track:
        hist["loglik"].append(mixture_loglik(Y, mask, params))
    for _ in range(n_iter):
        states, resp = mixture_estep(Y, mask, params, states, method, mean_sweeps, cov_sweeps)
        params = mixture_mstep(states, params, resp)
        if track:
            hist["loglik"].append(mixture_loglik(Y, mask, params))
    return params, resp, hist


def imputed_data(states, resp):
    """Y-hat_i = sum_g u-hat_ig Y-hat_ig, the pooled completion of Section 4.1."""
    stack = jnp.stack([st.Yhat for st in states], axis=1)
    return jnp.einsum("ng,ngpq->npq", resp, stack)


def adjusted_rand_index(a, b):
    """ARI (Hubert and Arabie, 1985), used to score the clustering."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    ka = int(jnp.max(a)) + 1
    kb = int(jnp.max(b)) + 1
    tab = jnp.zeros((ka, kb), dtype=jnp.float64).at[a, b].add(1.0)
    n = a.shape[0]

    def c2(x):
        return x * (x - 1.0) / 2.0

    sij = jnp.sum(c2(tab))
    si = jnp.sum(c2(jnp.sum(tab, axis=1)))
    sj = jnp.sum(c2(jnp.sum(tab, axis=0)))
    expected = si * sj / c2(float(n))
    maxi = 0.5 * (si + sj)
    return float((sij - expected) / (maxi - expected))
