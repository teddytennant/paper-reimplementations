"""Algorithm 1 / 2 invariants and DGP sanity checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dp_ate_blocking.dgp import well_specified
from dp_ate_blocking.estimators import bps, difference_in_means, ipw
from dp_ate_blocking.privacy import default_m, dp_blocking, sample_laplace, seq_ipw
from dp_ate_blocking.propensity import fit_propensity, predict_propensity


def test_seq_ipw_noise_free_equals_clipped_ipw():
    y = jnp.array([1.0, 0.5, -0.2, 1.2])
    w = jnp.array([1.0, 0.0, 1.0, 0.0])
    e = jnp.array([0.2, 0.8, 0.05, 0.97])
    x = jnp.zeros((4, 1))
    omega = 0.1
    got = seq_ipw(x, w, y, e_hat=e, omega=omega, add_noise=False)
    expect = ipw(y, w, e, omega=omega)
    assert abs(float(got) - float(expect)) < 1e-15


def test_dp_blocking_zero_noise_equals_bps():
    # Every equal-spaced bin has both treatments, so ñ = n and the
    # Algorithm 2 reduction matches BPS §2.2 (empty cells skipped the same way).
    y = jnp.array([2.0, 0.0, 4.0, 2.0, 1.0, -1.0])
    w = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    e = jnp.array([0.1, 0.2, 0.6, 0.7, 0.85, 0.9])
    x = jnp.zeros((6, 1))
    m = 2
    got = dp_blocking(x, w, y, e_hat=e, m=m, T=0.0, add_noise=False)
    expect = bps(y, w, e, m=m)
    assert abs(float(got) - float(expect)) < 1e-12


def test_dp_blocking_drops_bin_below_T():
    # m=2, bin 0 mixed and large; bin 1 has n_{k,1}=1 < T=1.5.
    y = jnp.array([3.0, 1.0, 2.0, 0.0, 5.0, -1.0])
    w = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    e = jnp.array([0.1, 0.15, 0.2, 0.25, 0.8, 0.85])
    x = jnp.zeros((6, 1))
    T = 1.5
    got = float(dp_blocking(x, w, y, e_hat=e, m=2, T=T, add_noise=False, C_y=5.0))
    # Only bin 0 is kept: n0=2, n1=2, s1=3+2=5, s0=1+0=1
    # tau_k = 4 * (5/2 - 1/2) = 8, ñ = 4, tau = 2
    assert abs(got - 2.0) < 1e-12
    # Dropped bin must not enter ñ. Full n is 6; if it were used as the
    # denominator the estimate would be 8/6 instead of 8/4.
    assert abs(got - 8.0 / 6.0) > 1e-6


def test_dp_blocking_all_dropped_returns_zero():
    y = jnp.array([1.0, 0.0])
    w = jnp.array([1.0, 0.0])
    e = jnp.array([0.1, 0.1])
    x = jnp.zeros((2, 1))
    got = float(dp_blocking(x, w, y, e_hat=e, m=2, T=10.0, add_noise=False))
    assert got == 0.0


def test_laplace_mean_near_zero():
    key = jax.random.PRNGKey(0)
    draws = sample_laplace(key, (20000,), 1.0)
    mean = float(jnp.mean(draws))
    assert abs(mean) < 0.05


def test_well_specified_ipw_beats_difference_in_means():
    key = jax.random.PRNGKey(7)
    x, w, y, e_true = well_specified(key, n=2000, tau=2.0)
    dim = float(difference_in_means(y, w))
    tau_true_e = float(ipw(y, w, e_true, omega=0.1))
    theta = fit_propensity(x, w, add_noise=False)
    e_fit = predict_propensity(theta, x)
    tau_fit = float(ipw(y, w, e_fit, omega=0.1))
    err_dim = abs(dim - 2.0)
    err_true = abs(tau_true_e - 2.0)
    err_fit = abs(tau_fit - 2.0)
    assert err_true < err_dim
    assert err_fit < err_dim


def test_seq_ipw_and_dp_blocking_finite_and_deterministic():
    key = jax.random.PRNGKey(11)
    x, w, y, _ = well_specified(key, n=80, tau=2.0)
    k1 = jax.random.PRNGKey(23)
    k2 = jax.random.PRNGKey(23)
    a = seq_ipw(x, w, y, key=k1, eps=1.0, C_y=1.5)
    b = seq_ipw(x, w, y, key=k2, eps=1.0, C_y=1.5)
    assert jnp.isfinite(a)
    assert float(a) == float(b)
    c = dp_blocking(x, w, y, key=k1, eps=1.0, C_y=1.5, m=3)
    d = dp_blocking(x, w, y, key=k2, eps=1.0, C_y=1.5, m=3)
    assert jnp.isfinite(c)
    assert float(c) == float(d)


def test_default_m_matches_paper_rule():
    # n=400, eps=1 -> inner = (0.4 + 3)/2 = 1.7 -> max(3,1)=3, cap=20 -> 3
    assert default_m(400, 1.0) == 3
    # large n, small eps: cap at max(5, 20ε)=5
    assert default_m(100_000, 0.1) == 5
    # large n and eps=1: inner=(100+3)/2=51.5 -> 51, cap=20 -> 20
    assert default_m(100_000, 1.0) == 20
