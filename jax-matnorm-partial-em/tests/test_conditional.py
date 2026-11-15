"""The E-step: exact conditional moments, and what the partial sweeps converge to."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matnorm_pem as mp
from matnorm_pem import synth


def _setup(p, q, rate, n=6, seed=0):
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(seed), params, n)
    mask = synth.mcar_mask(jr.key(seed + 100), n, p, q, rate)
    # keep at least one cell of each kind so the partitions are non-trivial
    mask = mask.at[:, 0, 0].set(True).at[:, -1, -1].set(False)
    return params, Y, mask


@pytest.mark.parametrize("p,q,rate", [(3, 4, 0.3), (4, 5, 0.5), (2, 3, 0.6)])
def test_exact_estep_matches_index_partition(p, q, rate):
    """The precision form (3) reproduces the textbook partition (2).

    (2) is written with Sigma_zx Sigma_xx^{-1}; the implementation uses
    Sigma_{z|x} = Xi_zz^{-1} and Sigma_zx Sigma_xx^{-1} = -Xi_zz^{-1} Xi_zx
    instead. Here the o x o partition is built by hand in numpy and compared.
    """
    params, Y, mask = _setup(p, q, rate)
    state = mp.exact_estep(Y, mask, params)
    Sigma = np.asarray(mp.kron_cov(params.S1, params.S2, params.sigma2))
    mu = np.asarray(mp.vec(params.M))
    for i in range(Y.shape[0]):
        mv = np.asarray(mp.vec(mask[i])).astype(bool)
        y = np.asarray(mp.vec(Y[i]))
        z = np.where(mv)[0]
        x = np.where(~mv)[0]
        B = Sigma[np.ix_(z, x)] @ np.linalg.inv(Sigma[np.ix_(x, x)])
        zmean = mu[z] + B @ (y[x] - mu[x])
        zcov = Sigma[np.ix_(z, z)] - B @ Sigma[np.ix_(x, z)]
        got_mean = np.asarray(mp.vec(state.Yhat[i]))
        got_cov = np.asarray(state.V[i])
        assert np.max(np.abs(got_mean[z] - zmean)) < 1e-10
        assert np.max(np.abs(got_mean[x] - y[x])) < 1e-12
        assert np.max(np.abs(got_cov[np.ix_(z, z)] - zcov)) < 1e-10
        assert np.max(np.abs(got_cov[np.ix_(x, x)])) < 1e-14


def test_padded_and_per_pattern_paths_agree():
    """The batched E-step and the literal per-pattern one give the same numbers."""
    params, Y, mask = _setup(4, 5, 0.4, n=5, seed=2)
    a = mp.exact_estep(Y, mask, params)
    b = mp.exact_estep_by_pattern(Y, mask, params)
    assert float(jnp.max(jnp.abs(a.Yhat - b.Yhat))) < 1e-12
    assert float(jnp.max(jnp.abs(a.V - b.V))) < 1e-12


def test_conditional_moments_match_monte_carlo():
    """Check the conditional mean and covariance against simulation.

    Drawing from the conditional directly would reuse the formula under test, so
    instead 120k joint draws are regressed: the least squares coefficient of z on
    x converges to Sigma_zx Sigma_xx^{-1} and the residual covariance to
    Sigma_{z|x}, neither of which touches the E-step code.
    """
    p, q, n = 2, 3, 120000
    params = synth.paper_params(p, q)
    mask = jnp.zeros((1, p, q), dtype=bool).at[0, 0, 0].set(True).at[0, 1, 2].set(True)
    Y = mp.sample_matnorm(jr.key(9), params, n)
    X = np.asarray(jax.vmap(mp.vec)(Y - params.M))
    mv = np.asarray(mp.vec(mask[0])).astype(bool)
    Z = X[:, mv]
    O = np.concatenate([X[:, ~mv], np.ones((n, 1))], axis=1)
    B, *_ = np.linalg.lstsq(O, Z, rcond=None)
    resid = Z - O @ B
    mc_cov = resid.T @ resid / n

    probe = mp.sample_matnorm(jr.key(21), params, 1)
    state = mp.exact_estep(probe, mask, params)
    analytic_mean = np.asarray(mp.vec(state.Yhat[0]))[mv]
    o = np.concatenate([np.asarray(mp.vec(probe[0] - params.M))[~mv], [1.0]])
    mc_mean = np.asarray(mp.vec(params.M))[mv] + o @ B
    analytic_cov = np.asarray(state.V[0])[np.ix_(mv, mv)]

    # standard error of each entry is around 0.005 at this sample size
    assert np.max(np.abs(analytic_mean - mc_mean)) < 0.03
    assert np.max(np.abs(analytic_cov - mc_cov)) < 0.03


@pytest.mark.parametrize("p,q,rate", [(3, 4, 0.3), (4, 5, 0.6)])
def test_partial_sweeps_converge_to_the_exact_moments(p, q, rate):
    """Section 3.1 and 3.2 claim the sweeps keep the exact E-step's fixed point.

    Repeated Gauss-Seidel sweeps of (6) minimize (5), which is the conditional
    mean, and repeated sweeps of (8) and (9) minimize (7), whose stationary
    point is Z-hat^{-1} = Xi_zz. Both are checked here against the exact E-step.
    """
    params, Y, mask = _setup(p, q, rate, n=6, seed=4)
    exact = mp.exact_estep(Y, mask, params)
    state = mp.init_estep_state(Y, mask, params)
    for _ in range(60):
        state = mp.partial_estep(Y, mask, params, state, mean_sweeps=1, cov_sweeps=1)
    assert float(jnp.max(jnp.abs(state.Yhat - exact.Yhat))) < 1e-9
    assert float(jnp.max(jnp.abs(state.V - exact.V))) < 1e-9


def test_one_partial_sweep_is_not_the_exact_answer():
    """The partial E-step really is partial; otherwise the tests above are hollow."""
    params, Y, mask = _setup(4, 5, 0.5, n=6, seed=6)
    exact = mp.exact_estep(Y, mask, params)
    state = mp.init_estep_state(Y, mask, params)
    state = mp.partial_estep(Y, mask, params, state, mean_sweeps=1, cov_sweeps=1)
    assert float(jnp.max(jnp.abs(state.V - exact.V))) > 1e-3


def test_covariance_sweeps_stay_positive_definite():
    """Section 3.2: each update sets the Schur complement to 1 / xi_w > 0."""
    params, Y, mask = _setup(4, 5, 0.7, n=6, seed=8)
    state = mp.init_estep_state(Y, mask, params)
    for _ in range(4):
        state = mp.partial_estep(Y, mask, params, state, cov_sweeps=1)
        for i in range(Y.shape[0]):
            mv = np.asarray(mp.vec(mask[i])).astype(bool)
            block = np.asarray(state.V[i])[np.ix_(mv, mv)]
            assert np.min(np.linalg.eigvalsh(block)) > 0


def test_free_energy_bounds_the_loglik_and_meets_it_at_convergence():
    """F <= observed log-likelihood, with equality when the KL divergence (4) is 0."""
    params, Y, mask = _setup(3, 4, 0.4, n=10, seed=12)
    ll = float(mp.observed_loglik(Y, mask, params))
    state = mp.init_estep_state(Y, mask, params)
    state = mp.partial_estep(Y, mask, params, state)
    assert float(mp.free_energy(Y, mask, params, state)) < ll
    for _ in range(60):
        state = mp.partial_estep(Y, mask, params, state)
    assert abs(float(mp.free_energy(Y, mask, params, state)) - ll) < 1e-8
