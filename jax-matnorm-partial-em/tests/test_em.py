"""Single-component EM: monotonicity, the headline claim, and recovery."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matnorm_pem as mp
from matnorm_pem import synth
from matnorm_pem.mstep import mstep


def _data(p, q, n, rate, seed):
    params = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(seed), params, n)
    mask = synth.mcar_mask(jr.key(seed + 500), n, p, q, rate)
    return params, Y, mask


def _min_increment(values):
    v = np.asarray(values, dtype=float)
    return float(np.min(np.diff(v)))


@pytest.mark.parametrize("rate", [0.15, 0.35, 0.6])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_em_never_decreases_the_loglik(rate, seed):
    """The EM guarantee itself. Nothing else in this file means much without it."""
    _, Y, mask = _data(3, 4, 120, rate, seed)
    _, hist = mp.fit_matnorm(Y, mask, method="exact", n_iter=30)
    scale = max(1.0, abs(hist["loglik"][-1]))
    assert _min_increment(hist["loglik"]) > -1e-9 * scale


@pytest.mark.parametrize("rate", [0.15, 0.35, 0.6])
@pytest.mark.parametrize("seed", [0, 1])
def test_partial_em_never_decreases_the_loglik(rate, seed):
    """Same for the partial E-step, scored on the exact observed log-likelihood.

    A partial E-step only guarantees the bound F goes up (see the next test);
    that the observed log-likelihood also goes up every iteration is the
    empirical claim, and it holds here to roundoff.
    """
    _, Y, mask = _data(3, 4, 120, rate, seed)
    _, hist = mp.fit_matnorm(Y, mask, method="partial", n_iter=30)
    scale = max(1.0, abs(hist["loglik"][-1]))
    assert _min_increment(hist["loglik"]) > -1e-9 * scale


@pytest.mark.parametrize("rate", [0.35, 0.6])
def test_partial_em_never_decreases_the_bound(rate):
    """F = loglik - KL of (4) is what a partial E-step provably increases."""
    _, Y, mask = _data(4, 5, 100, rate, 3)
    _, hist = mp.fit_matnorm(Y, mask, method="partial", n_iter=30)
    scale = max(1.0, abs(hist["free_energy"][-1]))
    assert _min_increment(hist["free_energy"]) > -1e-9 * scale
    for f, l in zip(hist["free_energy"], hist["loglik"]):
        assert f <= l + 1e-9 * scale


@pytest.mark.parametrize("p,q", [(3, 4), (4, 5)])
@pytest.mark.parametrize("rate", [0.1, 0.25, 0.5, 0.75])
def test_partial_em_reaches_the_same_loglik_as_exact_em(p, q, rate):
    """The headline claim of Section 3.5, "nearly the same observed likelihood".

    Two assertions. The absolute gap per observation stays under 1e-2, which is
    loose next to what is measured (below 1e-9 at 10 to 50 percent missing, a
    few times 1e-3 at 75 percent where 60 iterations have not fully converged
    the sweeps). And the gap is at least fifty times smaller than the plug-in
    method's gap on the same data, which is the comparison that matters: the
    covariance correction of (8) and (9) is doing real work, the agreement is
    not something any cheap imputation would give.
    """
    n = 150
    _, Y, mask = _data(p, q, n, rate, 7)
    _, h_exact = mp.fit_matnorm(Y, mask, method="exact", n_iter=60)
    _, h_partial = mp.fit_matnorm(Y, mask, method="partial", n_iter=60)
    _, h_plugin = mp.fit_matnorm(Y, mask, method="plugin", n_iter=60)
    gap = abs(h_exact["loglik"][-1] - h_partial["loglik"][-1]) / n
    plugin_gap = abs(h_exact["loglik"][-1] - h_plugin["loglik"][-1]) / n
    assert gap < 1e-2
    assert gap < plugin_gap / 50.0


@pytest.mark.parametrize("method", ["exact", "partial"])
def test_parameter_recovery(method):
    """Recover a known matrix normal from data with a quarter of the cells gone.

    Comparison is on the Section 2.4 normalized scale, |S1| = |S2| = 1 with the
    size in sigma^2, because the Kronecker factors are only identified up to a
    scalar and an unnormalized comparison would be meaningless. Errors are
    aggregated over five seeds so one unlucky sample cannot fail the test.
    """
    p, q, n = 3, 4, 400
    truth = synth.paper_params(p, q)
    errs = {"M": [], "S1": [], "S2": [], "sigma2": []}
    for seed in range(4):
        Y = mp.sample_matnorm(jr.key(seed), truth, n)
        mask = synth.mcar_mask(jr.key(seed + 900), n, p, q, 0.25)
        est, _ = mp.fit_matnorm(Y, mask, method=method, n_iter=50, track=False)
        errs["M"].append(float(jnp.max(jnp.abs(est.M - truth.M))))
        errs["S1"].append(float(jnp.max(jnp.abs(est.S1 - truth.S1))))
        errs["S2"].append(float(jnp.max(jnp.abs(est.S2 - truth.S2))))
        errs["sigma2"].append(abs(float(est.sigma2 - truth.sigma2)))
    assert np.mean(errs["M"]) < 0.12
    assert np.mean(errs["S1"]) < 0.12
    assert np.mean(errs["S2"]) < 0.12
    assert np.mean(errs["sigma2"]) < 0.06


def test_mstep_fixes_the_truth_when_everything_is_missing():
    """A tight algebraic check on the projections (11) and (12).

    With every cell missing, the exact E-step returns Y-hat = M and
    V-hat = sigma^2 (S2 (x) S1). The row projection then has to give
    R-hat = sigma^2 q S1 and the column projection C-hat = sigma^2 p S2, so the
    M-step must hand back exactly the parameters it was given. Any index slip in
    the (q, p, q, p) reshape shows up here immediately.
    """
    p, q = 3, 4
    truth = synth.paper_params(p, q)
    Y = jnp.zeros((5, p, q))
    mask = jnp.ones((5, p, q), dtype=bool)
    state = mp.exact_estep(Y, mask, truth)
    out = mstep(state, truth)
    assert float(jnp.max(jnp.abs(out.M - truth.M))) < 1e-12
    assert float(jnp.max(jnp.abs(out.S1 - truth.S1))) < 1e-12
    assert float(jnp.max(jnp.abs(out.S2 - truth.S2))) < 1e-12
    assert abs(float(out.sigma2 - truth.sigma2)) < 1e-12


def test_no_missing_data_converges_to_the_flip_flop_mle():
    """With complete data the M-step is the usual flip-flop and one fit is enough."""
    p, q, n = 3, 4, 300
    truth = synth.paper_params(p, q)
    Y = mp.sample_matnorm(jr.key(2), truth, n)
    mask = jnp.zeros((n, p, q), dtype=bool)
    est, hist = mp.fit_matnorm(Y, mask, method="exact", n_iter=25)
    assert _min_increment(hist["loglik"]) > -1e-9 * abs(hist["loglik"][-1])
    assert float(jnp.max(jnp.abs(est.M - jnp.mean(Y, axis=0)))) < 1e-12
    assert float(jnp.max(jnp.abs(est.S1 - truth.S1))) < 0.15


@pytest.mark.parametrize("method", ["exact", "partial"])
def test_values_hidden_behind_the_mask_are_never_read(method):
    """Nothing leaks from a masked cell into the fit.

    Every routine sanitizes with where(mask, 0, Y) before touching the data, so
    replacing the deleted cells with NaN or with 1e6 has to leave the fit bit
    for bit unchanged. Without this the imputation results would be quietly
    self-fulfilling.
    """
    _, Y, mask = _data(3, 4, 100, 0.3, 21)
    ref, _ = mp.fit_matnorm(Y, mask, method=method, n_iter=20, track=False)
    for filler in (jnp.nan, 1e6):
        Yx = jnp.where(mask, filler, Y)
        got, _ = mp.fit_matnorm(Yx, mask, method=method, n_iter=20, track=False)
        assert float(jnp.max(jnp.abs(got.M - ref.M))) == 0.0
        assert float(jnp.max(jnp.abs(got.S1 - ref.S1))) == 0.0
        assert float(jnp.max(jnp.abs(got.S2 - ref.S2))) == 0.0
        assert float(jnp.abs(got.sigma2 - ref.sigma2)) == 0.0
