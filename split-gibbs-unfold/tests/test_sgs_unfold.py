"""Lock displayed equations of arXiv:2609.30539 on a tiny CPU toy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sgs_unfold.likelihood import analytic_prox_g, analytic_unfold, gammas_from_params
from sgs_unfold.ode import (
    denoise_argument,
    denoise_noise_level,
    euler_reverse_step,
    ode_rhs,
)
from sgs_unfold.phase import fy, grad_fy, make_A, measurement_residual, observe
from sgs_unfold.prior import gaussian_denoiser, gaussian_score, make_prior_denoiser
from sgs_unfold.sampler import (
    decreasing_t_grid,
    l_step,
    p_step,
    split_gibbs_step,
    tau_schedule,
)
from sgs_unfold.schedule import (
    edm_sigma,
    edm_sigma_dot_over_sigma,
    integrate_beta,
    vp_sigma_dot_over_sigma,
    vp_sigma_sq,
    vp_t_from_sigma_sq,
)
from sgs_unfold.split import (
    coupling_kernel_log,
    g_tau_log_density_factor,
    lstep_tau,
    lstep_z,
    pstep_tau_z,
)
from sgs_unfold.likelihood import (
    init_unfolded,
    mse_denoise_loss,
    sample_surrogate_batch,
    train_unfolded,
    unfolded_denoise,
)


def test_tau_sigma_identity_from_beta_integral():
    """σ_k² = exp(∫_0^{τ_k} β_s ds) - 1 and the inverse inf{τ: ∫β = log(1+σ²)}."""
    beta = 0.35
    tau_k = jnp.array(1.4)
    integral = integrate_beta(beta, tau_k)
    np.testing.assert_allclose(float(integral), beta * float(tau_k), rtol=1e-6)
    sigma_sq = vp_sigma_sq(tau_k, beta)
    np.testing.assert_allclose(
        float(sigma_sq), float(np.exp(beta * float(tau_k)) - 1.0), rtol=1e-6
    )
    tau_back = vp_t_from_sigma_sq(sigma_sq, beta)
    np.testing.assert_allclose(float(tau_back), float(tau_k), rtol=1e-5)

    sigma_sq_k = jnp.array(2.25)
    tau_from_s = vp_t_from_sigma_sq(sigma_sq_k, beta)
    np.testing.assert_allclose(
        float(tau_from_s), float(np.log1p(2.25) / beta), rtol=1e-6
    )
    np.testing.assert_allclose(
        float(vp_sigma_sq(tau_from_s, beta)), 2.25, rtol=1e-5
    )


def test_eq6_denoise_arguments_and_rhs():
    """Displayed ODE (6): arguments and (σ̇/σ)(X̄ - D(·,·))."""
    xbar = jnp.array([1.0, -0.5, 0.25])
    z = jnp.array([0.2, 0.3, -0.1])
    tau = 0.4
    sigma = 1.5
    sdos = 1.0 / 1.5
    arg = denoise_argument(xbar, z, tau, sigma)
    eta = denoise_noise_level(tau, sigma)
    expected_arg = (tau**2 * np.array(xbar) + sigma**2 * np.array(z)) / (
        tau**2 + sigma**2
    )
    expected_eta = tau * sigma / np.sqrt(tau**2 + sigma**2)
    np.testing.assert_allclose(arg, expected_arg, rtol=1e-6)
    np.testing.assert_allclose(float(eta), expected_eta, rtol=1e-6)

    def fake_D(x, e):
        return 0.4 * x

    rhs = ode_rhs(xbar, z, tau, sigma, sdos, fake_D)
    expected_rhs = sdos * (np.array(xbar) - 0.4 * expected_arg)
    np.testing.assert_allclose(rhs, expected_rhs, rtol=1e-6)
    assert not np.allclose(rhs, 0.0)


def test_eq1_eq3_coupling_matches_g_tau_factor():
    """Eq. (1) coupling and eq. (3) Bayes factor are the same Gaussian kernel."""
    s = jnp.array([0.1, -0.2])
    s_prime = jnp.array([0.4, 0.0])
    tau = 0.7
    c = coupling_kernel_log(s, s_prime, tau)
    g = g_tau_log_density_factor(s, s_prime, tau)
    expected = -np.sum((np.array(s_prime) - np.array(s)) ** 2) / (2.0 * tau**2)
    np.testing.assert_allclose(float(c), expected, rtol=1e-6)
    np.testing.assert_allclose(float(g), expected, rtol=1e-6)


def test_eq5_lstep_tau_z_and_pstep_identity():
    """Eq. (5): τ_L = τ0 τ_i / √(τ0²-τ_i²), z = τ0² S_i / (τ0²-τ_i²)."""
    tau_0 = 1.0
    tau_i = 0.3
    s_i = jnp.array([1.2, -0.4, 0.5])
    s_prev = jnp.array([0.0, 1.0, -1.0])
    tau_p, z_p = pstep_tau_z(s_prev, tau_i)
    np.testing.assert_allclose(float(tau_p), tau_i)
    np.testing.assert_allclose(z_p, s_prev)
    tau_L = lstep_tau(tau_0, tau_i)
    z_L = lstep_z(s_i, tau_0, tau_i)
    np.testing.assert_allclose(
        float(tau_L), tau_0 * tau_i / np.sqrt(tau_0**2 - tau_i**2), rtol=1e-6
    )
    np.testing.assert_allclose(
        z_L, (tau_0**2) * np.array(s_i) / (tau_0**2 - tau_i**2), rtol=1e-6
    )
    assert float(tau_L) > 0.0


def test_ode_step_reduces_noise_gaussian_prior():
    """One reverse Euler step of eq. (6) with analytic Gaussian denoiser."""
    prior_var = 1.0
    denoiser = make_prior_denoiser(prior_var)
    z = jnp.zeros((8,))
    tau = 0.5
    t = 2.0
    dt = -0.4
    sigma = edm_sigma(t)
    sdos = edm_sigma_dot_over_sigma(t)
    key = jax.random.PRNGKey(0)
    xbar = sigma * jax.random.normal(key, (8,))
    x_next = euler_reverse_step(xbar, z, tau, sigma, sdos, dt, denoiser)
    # Reverse flow with z=0 contracts toward the origin (posterior mean).
    assert float(jnp.linalg.norm(x_next)) < float(jnp.linalg.norm(xbar))
    score = gaussian_score(xbar, 0.8, prior_var)
    expected_score = -np.array(xbar) / (prior_var + 0.8**2)
    np.testing.assert_allclose(score, expected_score, rtol=1e-5)
    d = gaussian_denoiser(xbar, 0.8, prior_var)
    np.testing.assert_allclose(d, np.array(xbar) / (prior_var + 0.8**2), rtol=1e-6)


def test_eq7_analytic_prox_and_grad_fy():
    """Eq. (7) closed-form prox plus ∇f_y vs jax.grad of f_y."""
    v = jnp.array([0.5, -0.2, 0.1, 0.0])
    x = jnp.array([0.4, 0.0, -0.3, 0.2])
    eta = 0.6
    tau_0 = 1.0
    gamma = 0.08
    prox = analytic_prox_g(v, x, eta, tau_0, gamma)
    num = np.array(v) + (gamma / eta**2) * np.array(x)
    den = 1.0 + gamma / tau_0**2 + gamma / eta**2
    np.testing.assert_allclose(prox, num / den, rtol=1e-6)

    key = jax.random.PRNGKey(1)
    A = make_A(key, 8, 4)
    s = jnp.array([0.3, -0.1, 0.2, 0.4])
    y = jnp.abs(A @ s) + 0.01
    meas_std = 0.2
    g_auto = jax.grad(fy)(s, y, A, meas_std)
    g_closed = grad_fy(s, y, A, meas_std)
    np.testing.assert_allclose(g_closed, g_auto, rtol=1e-4, atol=1e-5)


def test_unfolded_likelihood_decreases_residual_vs_doing_nothing():
    """Analytic unfold of eq. (7) lowers ||y-|A s|| vs the noisy input."""
    n, m = 8, 16
    key = jax.random.PRNGKey(3)
    kA, k_s, k_y, k_n = jax.random.split(key, 4)
    A = make_A(kA, m, n)
    tau_0 = 1.0
    meas_std = 0.2
    s_true = tau_0 * jax.random.normal(k_s, (n,))
    y = observe(k_y, A, s_true, meas_std)
    eta = 0.3
    x_noisy = s_true + eta * jax.random.normal(k_n, (n,))
    lip = jnp.linalg.norm(A, ord=2) ** 2 / (meas_std**2)
    gamma = 0.5 / lip
    gammas = jnp.array([gamma, gamma, gamma])
    s_hat = analytic_unfold(x_noisy, eta, y, A, tau_0, meas_std, gammas)
    r_hat = measurement_residual(s_hat, y, A)
    r_noop = measurement_residual(x_noisy, y, A)
    assert float(r_hat) < float(r_noop)
    assert s_hat.shape == x_noisy.shape


def test_trained_unfold_not_identity_and_pytree_params():
    n, m = 8, 16
    key = jax.random.PRNGKey(4)
    kA, k_tr, k_b = jax.random.split(key, 3)
    A = make_A(kA, m, n)
    tau_0 = 1.0
    meas_std = 0.2
    params, losses = train_unfolded(
        k_tr,
        A,
        tau_0,
        meas_std,
        steps=8,
        batch=16,
        lr=5e-3,
        n_layers=2,
        hidden=8,
    )
    leaves = jax.tree_util.tree_leaves(params)
    assert len(leaves) >= 2
    assert all(hasattr(leaf, "shape") for leaf in leaves)
    gammas = gammas_from_params(params)
    assert jnp.all(gammas > 0)
    assert losses.shape == (8,)
    s, y, eta, x = sample_surrogate_batch(
        k_b, A, tau_0, meas_std, batch=4, eta_min=0.1, eta_max=0.5
    )
    pred = jax.vmap(
        lambda xi, ei, yi: unfolded_denoise(
            params, xi, ei, yi, A, tau_0, meas_std
        )
    )(x, eta, y)
    assert pred.shape == s.shape
    assert not jnp.allclose(pred, x, atol=1e-6)
    loss = mse_denoise_loss(params, s, y, eta, x, A, tau_0, meas_std)
    assert float(loss) >= 0.0


def test_prior_and_likelihood_are_alternating_denoisers_not_joint_mcmc():
    """P-step ignores y; L-step uses y; one outer step is P then L, not MH."""
    n, m = 8, 16
    key = jax.random.PRNGKey(5)
    kA, k_s, k_y, k_p, k_run = jax.random.split(key, 5)
    A = make_A(kA, m, n)
    tau_0 = 1.0
    tau_i = 0.4
    meas_std = 0.2
    prior_var = tau_0**2
    s_prime = jax.random.normal(k_s, (n,))
    y1 = observe(k_y, A, s_prime, meas_std)
    y2 = y1 + 1.5
    t_grid = decreasing_t_grid(t_max=1.0, t_min=0.2, n_steps=3)
    D_P = make_prior_denoiser(prior_var)
    params = init_unfolded(k_p, n_layers=2, hidden=4)
    D_L1 = lambda x, eta, y=y1: unfolded_denoise(
        params, x, eta, y, A, tau_0, meas_std
    )
    D_L2 = lambda x, eta, y=y2: unfolded_denoise(
        params, x, eta, y, A, tau_0, meas_std
    )

    k_p, k_l = jax.random.split(k_run)
    s_from_p = p_step(
        k_p, s_prime, tau_i, D_P, t_grid, edm_sigma, edm_sigma_dot_over_sigma
    )
    # Same key, same s': P-step cannot depend on y.
    s_from_p_again = p_step(
        k_p, s_prime, tau_i, D_P, t_grid, edm_sigma, edm_sigma_dot_over_sigma
    )
    np.testing.assert_allclose(s_from_p, s_from_p_again, rtol=1e-6)

    sp1 = l_step(
        k_l, s_from_p, tau_i, tau_0, D_L1, t_grid, edm_sigma, edm_sigma_dot_over_sigma
    )
    sp2 = l_step(
        k_l, s_from_p, tau_i, tau_0, D_L2, t_grid, edm_sigma, edm_sigma_dot_over_sigma
    )
    assert not np.allclose(sp1, sp2, atol=1e-5)

    s_alt, sp_alt = split_gibbs_step(
        k_run,
        s_prime,
        tau_i,
        tau_0,
        D_P,
        D_L1,
        t_grid,
        edm_sigma,
        edm_sigma_dot_over_sigma,
    )
    np.testing.assert_allclose(s_alt, s_from_p, rtol=1e-6)
    np.testing.assert_allclose(sp_alt, sp1, rtol=1e-5)
    assert s_alt.shape == (n,)
    assert sp_alt.shape == (n,)
    taus = tau_schedule(0.5, 0.2, 4)
    assert taus.shape == (4,)
    assert jnp.all(taus > 0)
    assert jnp.all(taus < tau_0)
    assert float(taus[0]) > float(taus[-1])


def test_vp_sigma_dot_over_sigma_matches_identity():
    beta = 0.5
    t = jnp.array(0.8)
    s2 = vp_sigma_sq(t, beta)
    sdos = vp_sigma_dot_over_sigma(t, beta)
    expected = beta * (s2 + 1.0) / (2.0 * s2)
    np.testing.assert_allclose(float(sdos), float(expected), rtol=1e-6)
    assert float(s2) > 0.0


def test_init_unfolded_nonnegative_gammas_and_shapes():
    key = jax.random.PRNGKey(9)
    params = init_unfolded(key, n_layers=2, hidden=8)
    gammas = gammas_from_params(params)
    np.testing.assert_allclose(np.array(gammas), np.array(gammas).clip(min=0))
    assert jnp.all(gammas > 0)
    n, m = 8, 16
    A = make_A(jax.random.PRNGKey(0), m, n)
    x = jnp.ones((n,))
    y = jnp.ones((m,))
    out = unfolded_denoise(params, x, 0.3, y, A, 1.0, 0.2)
    assert out.shape == (n,)
    assert jnp.all(jnp.isfinite(out))
