"""Checks for the diffusion coefficient, compositional score, HBDS, and path loss."""

import jax
import jax.numpy as jnp
import pytest

from hbds.diffusion import (
    beta_schedule,
    g_n_squared,
    g_n_squared_from_variance_ode,
    g_ratio_from_alpha,
    marginal_variance,
    marginal_variance_quadrature,
    probability_flow_step,
    reverse_drift,
)
from hbds.gaussian import (
    analytic_posterior,
    analytic_posterior_from_factors,
    compositional_noised_score,
    compositional_score,
    factorized_noised_score,
    factorized_posterior_score,
    log_noised_density,
    log_posterior_density,
    sample_posterior,
    simulate_observations,
)
from hbds.hbds import (
    grouped_sign_posterior_mass,
    hbds_sample,
    make_grouped_observations,
    pooled_sample,
    sign_agreement,
)
from hbds.path import (
    dsm_nll,
    mlp_param_count,
    observation_score,
    path_divergence,
    path_divergence_from_values,
    path_regularized_loss,
    simulator_score,
)


def test_g1_equals_beta():
    ts = jnp.linspace(1e-4, 1.0, 25)
    err = jnp.max(jnp.abs(g_n_squared(ts, 1) - beta_schedule(ts)))
    assert float(err) < 1e-10


def test_gn_bounded_by_beta():
    ts = jnp.linspace(0.0, 1.0, 21)
    for n in (1, 2, 4, 8, 16):
        slack = jnp.min(beta_schedule(ts) - g_n_squared(ts, n))
        assert float(slack) >= -1e-10


def test_gn_high_noise_limit():
    # alpha_bar(1) is about 4e-5 under the default schedule, so t = 1 is the small-alpha check.
    for n in (1, 2, 3, 8, 16):
        ratio = g_n_squared(1.0, n) / beta_schedule(1.0)
        assert abs(float(ratio) - 1.0 / n) < 1e-4
        closed = g_ratio_from_alpha(0.0, n)
        assert abs(float(closed) - 1.0 / n) < 1e-12


def test_variance_quadrature_matches_closed_form():
    for t in (0.25, 0.5, 0.75, 1.0):
        for n in (1, 3, 8):
            closed = float(marginal_variance(t, n))
            quad = float(marginal_variance_quadrature(t, n, n_grid=512))
            rel = abs(quad - closed) / abs(closed)
            assert rel < 1e-3, f"t={t} n={n} rel={rel}"


def test_diffusion_coefficient_matches_variance_ode():
    for t in (0.2, 0.5, 0.8):
        for n in (1, 4, 8):
            closed = float(g_n_squared(t, n))
            ode = float(g_n_squared_from_variance_ode(t, n))
            rel = abs(ode - closed) / abs(closed)
            assert rel < 1e-3, f"t={t} n={n} rel={rel}"


def test_reverse_drift_matches_equation():
    state = jnp.array([0.3, -0.2])
    score = jnp.array([1.0, -0.5])
    t = 0.4
    n = 5
    got = reverse_drift(state, t, score, n)
    expect = 0.5 * beta_schedule(t) * state + g_n_squared(t, n) * score
    assert jnp.allclose(got, expect)


def test_probability_flow_uses_half_score_coefficient():
    state = jnp.array([0.2])
    score = jnp.array([0.5])
    t = 0.3
    dt = 0.01
    n = 2
    stepped = probability_flow_step(state, t, dt, score, n)
    drift = 0.5 * beta_schedule(t) * state + 0.5 * g_n_squared(t, n) * score
    assert jnp.allclose(stepped, state + drift * dt)


def _theta_grid():
    coords = jnp.linspace(-1.2, 1.2, 5)
    xx, yy = jnp.meshgrid(coords, coords, indexing="ij")
    return jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def test_score_identity_clean_and_noised():
    """Compositional score equals the closed-form posterior score at each noise level.

    Clean level: s_prior + sum of likelihood scores, the factorized sum of
    per-observation posterior scores, and jax.grad of log(prior * product of
    likelihoods) agree on a grid. No Jacobian term is added.

    Noised level: the same product posterior, pushed through the VP kernel, has
    one score whether the posterior was built by summing likelihood precisions
    or by the (1-n) prior correction. That score matches jax.grad of the noised
    Gaussian log density.
    """
    xis, ys, _ = simulate_observations(jax.random.PRNGKey(0))
    mean_sum, cov_sum = analytic_posterior(ys, xis)
    mean_fac, cov_fac = analytic_posterior_from_factors(ys, xis)
    assert jnp.allclose(mean_sum, mean_fac, atol=1e-8)
    assert jnp.allclose(cov_sum, cov_fac, atol=1e-8)

    grad_clean = jax.grad(log_posterior_density)
    for theta in _theta_grid():
        s_comp = compositional_score(theta, ys, xis)
        s_fact = factorized_posterior_score(theta, ys, xis)
        s_grad = grad_clean(theta, ys, xis)
        assert jnp.allclose(s_comp, s_grad, atol=1e-6, rtol=1e-6)
        assert jnp.allclose(s_fact, s_grad, atol=1e-6, rtol=1e-6)

    n = ys.shape[0]
    for t in (0.05, 0.2, 0.5, 0.9):
        grad_noised = jax.grad(lambda th: log_noised_density(th, t, mean_sum, cov_sum, n))
        for theta in _theta_grid():
            s_comp = compositional_noised_score(theta, t, ys, xis)
            s_fact = factorized_noised_score(theta, t, ys, xis)
            s_grad = grad_noised(theta)
            assert jnp.allclose(s_comp, s_fact, atol=1e-6, rtol=1e-6)
            assert jnp.allclose(s_comp, s_grad, atol=1e-5, rtol=1e-5)


def test_reverse_sde_matches_posterior_mean():
    xis, ys, _ = simulate_observations(jax.random.PRNGKey(0))
    samples, mean, x0 = sample_posterior(
        ys,
        xis,
        n_samples=128,
        n_steps=32,
        key=jax.random.PRNGKey(1),
    )
    start_err = float(jnp.linalg.norm(jnp.mean(x0, axis=0) - mean))
    end_err = float(jnp.linalg.norm(jnp.mean(samples, axis=0) - mean))
    # The chain must move: the terminal law is near 0, the posterior mean is not.
    assert start_err > 0.5
    assert end_err < 0.25, f"mean error {end_err} sample mean {jnp.mean(samples, axis=0)} target {mean}"


def test_hbds_sign_agreement_beats_pooled():
    observations, signs, _theta = make_grouped_observations(
        jax.random.PRNGKey(0), n_per_group=16, signs=jnp.array([1.0, -1.0])
    )
    mass = float(grouped_sign_posterior_mass(observations, signs))
    assert mass > 0.8
    assert float(signs[0] * signs[1]) < 0

    key = jax.random.PRNGKey(1)
    _theta_g, z_g = hbds_sample(observations, n_samples=48, n_steps=40, key=key)
    _theta_p, z_p = pooled_sample(
        observations.reshape(-1), n_samples=48, n_steps=40, key=key
    )
    agree_g = float(sign_agreement(z_g, signs))
    z_tied = jnp.stack([z_p, z_p], axis=1)
    agree_p = float(sign_agreement(z_tied, signs))
    assert agree_g > agree_p
    assert agree_g > 0.5


def test_hbds_follows_same_sign_data():
    """A sampler that hardcodes opposite signs fails when both groups are positive."""
    signs = jnp.array([1.0, 1.0])
    observations, signs, _theta = make_grouped_observations(
        jax.random.PRNGKey(2), n_per_group=16, signs=signs
    )
    mass = float(grouped_sign_posterior_mass(observations, signs))
    assert mass > 0.8
    _theta_g, z_g = hbds_sample(observations, n_samples=32, n_steps=40, key=jax.random.PRNGKey(3))
    agree = float(sign_agreement(z_g, signs))
    assert agree > 0.5


def test_path_divergence_zero_when_scores_match():
    times = jnp.linspace(0.05, 1.0, 12)
    states = jnp.linspace(-0.5, 0.5, 12)[:, None]

    def score(state, t, cond):
        return state * t + cond

    a = path_divergence(score, score, states, times, cond=jnp.array([0.25]))
    assert float(a) == pytest.approx(0.0, abs=1e-12)


def test_path_divergence_positive_when_scores_differ():
    times = jnp.linspace(0.05, 1.0, 12)
    states = jnp.zeros((12, 1))

    def left(state, t, cond):
        return jnp.zeros_like(state)

    def right(state, t, cond):
        return jnp.ones_like(state)

    a = path_divergence(left, right, states, times, cond=None, n=1)
    manual = path_divergence_from_values(jnp.zeros(12), jnp.ones(12), times, n=1)
    assert float(a) > 0.0
    assert float(a) == pytest.approx(float(manual), rel=1e-10)


def test_path_regularized_loss_adds_terms():
    nll = jnp.array(1.25)
    path_a = jnp.array(0.4)
    lam = 0.3
    total = path_regularized_loss(nll, path_a, lam)
    assert float(total) == pytest.approx(1.25 + 0.3 * 0.4)

    # Identical simulator and fine-tune scores: the path term vanishes, so L = L_NLL.
    y = jnp.array([0.2, -0.4, 0.1])
    t = jnp.array([0.2, 0.5, 0.8])
    theta = jnp.array([0.3, -0.1, 0.7])
    xi = jnp.array([0.5, -0.2, 0.0])
    s_pt = simulator_score(y, t, theta, xi)
    s_obs = observation_score(y, t, theta, xi)
    nll_exact = dsm_nll(s_pt, s_pt)
    assert float(nll_exact) == pytest.approx(0.0, abs=1e-12)
    # Observation score differs from the simulator score when the bias is nonzero.
    assert float(dsm_nll(s_obs, s_pt)) > 0.0
    total_zero_path = path_regularized_loss(dsm_nll(s_obs, s_obs), jnp.array(0.0), 1.0)
    assert float(total_zero_path) == pytest.approx(0.0, abs=1e-12)


def test_score_mlp_is_small():
    n_params = mlp_param_count()
    assert 100 < n_params < 800
