"""The estimation half of the paper: Algorithm 1, parameter recovery, and the
claim that skewness buys something when the truth is skewed.

Every assertion here is an aggregate over seeds with a stated margin. Section 3.6
is explicit that at M = 1 this is stochastic EM and the observed log-likelihood
is not monotone, so per-iteration monotonicity is not asserted anywhere.
"""

import jax
import numpy as np

from dstmm import (
    init_from_data,
    mcem_fit,
    paper_style_truth,
    per_point_log_likelihood,
    random_init,
    sample_dstmm,
)

SEEDS = (0, 1, 2)


def test_log_likelihood_trends_up_from_a_cold_start():
    """Equation (11) evaluated along the run has to climb, judged on a window mean
    rather than iteration by iteration."""
    gains = []
    for seed in SEEDS:
        truth = paper_style_truth(kappa=1.5, nu=6.0)
        k = jax.random.split(jax.random.key(200 + seed), 3)
        y, _, _, _ = sample_dstmm(k[0], truth, 300)
        init = random_init(k[1], truth.dims, truth.ks, active_set=(1,), nu0=10.0)
        res = mcem_fit(k[2], init, y, n_iter=24, active_set=(1,), n_mc=2)
        trace = np.asarray(res.log_lik)
        assert np.all(np.isfinite(trace))
        early = trace[1:6].mean()
        late = trace[-5:].mean()
        gains.append(late - early)
        assert late > early
    assert np.mean(gains) > 20.0


def test_fitted_model_beats_its_own_starting_point():
    """A weaker but sharper statement: the fit ends above where it started, every seed."""
    for seed in SEEDS:
        truth = paper_style_truth(kappa=1.0, nu=6.0)
        k = jax.random.split(jax.random.key(300 + seed), 3)
        y, _, _, _ = sample_dstmm(k[0], truth, 250)
        init = random_init(k[1], truth.dims, truth.ks, active_set=(1,), nu0=10.0)
        res = mcem_fit(k[2], init, y, n_iter=15, active_set=(1,), n_mc=2)
        assert float(res.log_lik[-1]) > float(res.log_lik[0]) + 50.0


def test_parameter_recovery():
    """Recovery of the common nu and of the first-layer skewness vectors on data
    drawn from a known DStMM, warm-started from the symmetric delta = 0 fit the
    way Section 3.6 prescribes.

    RMSE_delta is the paper's Section 4.2 definition,
    sqrt( (1/(2p)) sum_a || delta_hat_a - delta_a ||^2 ). The generator gives both
    first-layer components the same delta = kappa v, so label switching at layer 1
    leaves this quantity alone. The null value, for delta_hat = 0, is
    sqrt(2 kappa^2 / (2p)) = kappa / sqrt(p) = 0.816 at kappa = 2, p = 6.
    """
    nu_true, kappa = 6.0, 2.0
    rmses, nus = [], []
    for seed in SEEDS:
        truth = paper_style_truth(kappa=kappa, nu=nu_true)
        k = jax.random.split(jax.random.key(100 + seed), 4)
        y, _, _, _ = sample_dstmm(k[0], truth, 600)
        init = init_from_data(k[1], y, truth.dims, truth.ks, active_set=(), nu0=10.0)
        symmetric = mcem_fit(k[2], init, y, n_iter=16, active_set=(), n_mc=2)
        fit = mcem_fit(k[3], symmetric.model, y, n_iter=24, active_set=(1,), n_mc=2)

        d_hat = np.asarray(fit.model.delta[0])
        d_true = np.asarray(truth.delta[0])
        rmses.append(np.sqrt(((d_hat - d_true) ** 2).sum() / (2 * truth.dims[0])))
        nus.append(float(fit.model.nu[0]))

    null_rmse = kappa / np.sqrt(6)
    assert np.mean(rmses) < 0.6 * null_rmse
    assert max(rmses) < 0.75 * null_rmse
    assert abs(np.mean(nus) - nu_true) < 0.7
    assert 4.5 < min(nus) and max(nus) < 8.0


def _held_out_gap(seed, kappa, nu=5.0, n=300, warm=14, extra=14):
    """Mean held-out log-likelihood of the skewed fit minus the symmetric one.

    Both continuations start from the same symmetric warm fit and are handed the
    same MCEM key, so they see identical latent draws wherever their parameters
    agree. That pairing removes most of the run-to-run Monte Carlo noise and
    leaves the difference that is actually under test: whether delta_a^(1) is
    estimated or pinned at zero, which is Corollary 3 part 1.
    """
    truth = paper_style_truth(kappa=kappa, nu=nu)
    k = jax.random.split(jax.random.key(seed), 4)
    y_train, _, _, _ = sample_dstmm(k[0], truth, n)
    y_test, _, _, _ = sample_dstmm(k[1], truth, n)
    init = init_from_data(k[2], y_train, truth.dims, truth.ks, active_set=(), nu0=10.0)
    warm_fit = mcem_fit(k[3], init, y_train, n_iter=warm, active_set=(), n_mc=2)

    k2 = jax.random.key(seed + 1000)
    skewed = mcem_fit(k2, warm_fit.model, y_train, n_iter=extra, active_set=(1,), n_mc=2)
    symmetric = mcem_fit(k2, warm_fit.model, y_train, n_iter=extra, active_set=(), n_mc=2)
    return float(per_point_log_likelihood(skewed.model, y_test).mean()) - float(
        per_point_log_likelihood(symmetric.model, y_test).mean()
    )


def test_skewness_helps_when_the_truth_is_skewed():
    """Table 1's qualitative claim, restated as held-out log-likelihood.

    At kappa = 2 with nu = 5 the skewed model should win; at kappa = 0 the two
    should be within noise of each other, which is the nesting result the paper
    leads with. At this scale (p = 6, n = 300, 14 + 14 iterations) the measured
    gaps are about +0.09 nats per point with a between-seed standard error of
    0.02 at kappa = 2, and -0.02 +/- 0.03 at kappa = 0, so the thresholds below
    are set from that spread rather than from the paper's numbers, which come
    from p = 20, n = 1000 and 300 iterations.
    """
    seeds = range(5)
    skewed_gaps = np.array([_held_out_gap(s, kappa=2.0) for s in seeds])
    se = skewed_gaps.std(ddof=1) / np.sqrt(len(skewed_gaps))
    assert skewed_gaps.mean() > 0.03
    assert skewed_gaps.mean() / se > 2.5
    assert (skewed_gaps > 0).sum() >= 4

    symmetric_gaps = np.array([_held_out_gap(s, kappa=0.0) for s in seeds])
    assert abs(symmetric_gaps.mean()) < 0.1
    assert skewed_gaps.mean() > symmetric_gaps.mean() + 0.06
