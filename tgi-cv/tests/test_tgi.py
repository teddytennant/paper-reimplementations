import math
import os

import jax
import jax.numpy as jnp
import pytest

from tgi_cv import (
    THETA,
    acceptance_probability,
    affine_apply,
    affine_log_det,
    asymptotic_factor,
    beta_from_gamma,
    eigenvalue,
    evaluate_modes,
    g_value,
    gi_propose,
    hermite_multivariate,
    hermite_probabilists,
    log_w,
    log_w_affine,
    log_w_transform,
    lower_triangular,
    mode_eigenvalues,
    objective,
    poisson_eta,
    prune_mask,
    pushforward_gaussian_log_prob,
    qg_value,
    run_tgi_rwm,
    sample_tail,
    snis,
    summand,
    tail_penalty,
)
from tgi_cv._host import NIX_LD_LIB, ensure_nix_ld


def test_nix_ld_prepends_once(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    ensure_nix_ld()
    path = os.environ["LD_LIBRARY_PATH"]
    assert path.startswith(NIX_LD_LIB)
    assert "nix-ld" in path
    ensure_nix_ld()
    assert os.environ["LD_LIBRARY_PATH"].count(NIX_LD_LIB) == 1


def test_import_sets_nix_ld():
    assert "nix-ld" in os.environ.get("LD_LIBRARY_PATH", "")


def test_probabilists_recurrence():
    x = jnp.array([-1.3, 0.0, 0.4, 1.7])
    he0 = hermite_probabilists(0, x)
    he1 = hermite_probabilists(1, x)
    assert jnp.allclose(he0, 1.0)
    assert jnp.allclose(he1, x)
    he2 = hermite_probabilists(2, x)
    he3 = hermite_probabilists(3, x)
    he4 = hermite_probabilists(4, x)
    assert jnp.allclose(he2, x * he1 - 1 * he0)
    assert jnp.allclose(he3, x * he2 - 2 * he1)
    assert jnp.allclose(he4, x * he3 - 3 * he2)
    assert jnp.allclose(he2, x**2 - 1.0)
    assert jnp.allclose(he3, x**3 - 3.0 * x)
    assert jnp.allclose(he4, x**4 - 6.0 * x**2 + 3.0)


def test_multivariate_normalized_product():
    z = jnp.array([0.5, -1.2, 0.3])
    k = (2, 0, 1)
    expected = hermite_probabilists(2, z[0]) / math.sqrt(2.0)
    expected = expected * hermite_probabilists(0, z[1])
    expected = expected * hermite_probabilists(1, z[2]) / math.sqrt(1.0)
    assert jnp.allclose(hermite_multivariate(k, z), expected)
    assert jnp.allclose(hermite_multivariate((0, 0), z[:2]), 1.0)


def test_orthonormal_under_standard_normal():
    z = jax.random.normal(jax.random.PRNGKey(1), (80000, 2))
    modes = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (1, 2)]
    phi = evaluate_modes(modes, z)
    gram = (phi.T @ phi) / z.shape[0]
    assert jnp.allclose(gram, jnp.eye(len(modes)), atol=0.04)


def test_eigenvalue_is_beta_to_order():
    beta = 0.7
    k = (1, 2, 0)
    assert mode_order_sum(k) == 3
    assert jnp.allclose(eigenvalue(k, beta), beta**3)
    modes = [(1, 0), (0, 2), (2, 1)]
    lam = mode_eigenvalues(modes, beta)
    assert jnp.allclose(lam, jnp.array([beta, beta**2, beta**3]))


def mode_order_sum(k):
    return sum(k)


def test_gi_proposal_and_conditional_expectation():
    gamma = 0.25
    beta = 0.75
    assert jnp.allclose(beta_from_gamma(gamma), beta)
    z = jnp.array([0.4, -1.1, 0.2])
    xi = jax.random.normal(jax.random.PRNGKey(7), (50000, 3))
    z_prop, beta_hat = gi_propose(z, xi, gamma)
    assert jnp.allclose(beta_hat, beta)
    manual = beta * z + jnp.sqrt(1.0 - beta**2) * xi
    assert jnp.allclose(z_prop, manual)
    modes = [(1, 0, 0), (0, 2, 0), (1, 1, 0), (0, 0, 3), (2, 0, 1)]
    for k in modes:
        est = jnp.mean(hermite_multivariate(k, z_prop))
        truth = eigenvalue(k, beta) * hermite_multivariate(k, z)
        assert jnp.allclose(est, truth, atol=0.05, rtol=0.08), (k, float(est), float(truth))


def test_lag1_matches_eigenvalue():
    gamma = 0.3
    beta = beta_from_gamma(gamma)
    key_z, key_xi = jax.random.split(jax.random.PRNGKey(3))
    z = jax.random.normal(key_z, (30000, 2))
    xi = jax.random.normal(key_xi, (30000, 2))
    z_prop, _ = gi_propose(z, xi, gamma)
    h = hermite_multivariate((1, 1), z)
    h_prop = hermite_multivariate((1, 1), z_prop)
    corr = jnp.mean(h * h_prop)
    assert jnp.allclose(corr, beta ** 2, atol=0.03)


def test_empty_dictionary_factor():
    lam = jnp.array([0.0, 0.5, 0.8])
    got = asymptotic_factor(lam)
    assert jnp.allclose(got, jnp.array([1.0, 3.0, 9.0]))
    assert jnp.allclose(got, (1.0 + lam) / (1.0 - lam))


def test_log_w_formula_drops_rho_constant():
    z = jnp.array([0.3, -0.2, 0.5])
    log_pi_tz = jnp.array(-0.4)
    log_det = jnp.array(0.25)
    got = log_w(z, log_pi_tz, log_det)
    manual = log_pi_tz + log_det + 0.5 * jnp.sum(z**2)
    assert jnp.allclose(got, manual)
    rho_constant = 0.5 * z.shape[0] * jnp.log(2.0 * jnp.pi)
    assert jnp.abs(got - (manual - rho_constant)) > 1.0


def test_affine_positive_diagonal_and_log_det():
    raw = jnp.array([-0.4, 0.3, 1.2, -0.2, 0.5, -1.0])
    d = 3
    L = lower_triangular(raw, d)
    assert jnp.allclose(jnp.triu(L, k=1), 0.0)
    assert jnp.all(jnp.diag(L) > 0.0)
    assert jnp.allclose(jnp.diag(L), jnp.exp(jnp.array([-0.4, 1.2, -1.0])))
    assert jnp.allclose(affine_log_det(L), jnp.sum(jnp.log(jnp.diag(L))))
    sign, logabs = jnp.linalg.slogdet(L)
    assert jnp.allclose(affine_log_det(L), logabs)
    assert float(sign) == 1.0


def test_jacobian_log_det_matches_affine():
    raw = jnp.array([0.2, -0.5, -0.1])
    b = jnp.array([0.3, -0.7])
    params = {"raw_tril": raw, "b": b}
    L = lower_triangular(raw, 2)

    def transform(z):
        return L @ z + b

    z = jnp.array([0.4, -0.8])
    x, log_det, _ = affine_apply(z, params)
    assert jnp.allclose(x, transform(z))
    assert jnp.allclose(log_det, jnp.sum(jnp.log(jnp.diag(L))))

    def log_pi(x):
        return -jnp.sum(x**2)

    assert jnp.allclose(
        log_w_affine(z, params, log_pi),
        log_w_transform(z, transform, log_pi),
    )


def test_pushforward_log_w_constant_and_rwm_accepts_all():
    raw = jnp.array([0.15, -0.4, -0.25])
    b = jnp.array([0.4, -0.6])
    params = {"raw_tril": raw, "b": b}
    L = lower_triangular(raw, 2)
    log_pi = pushforward_gaussian_log_prob(L, b)
    z = jax.random.normal(jax.random.PRNGKey(4), (64, 2))
    lw = log_w_affine(z, params, log_pi)
    const = -0.5 * 2.0 * jnp.log(2.0 * jnp.pi)
    assert jnp.allclose(lw, const, atol=1e-8)
    assert float(jnp.std(lw)) < 1e-8

    def log_w_fn(z1):
        return jnp.zeros((), dtype=z1.dtype)

    def f_fn(z1):
        return z1[0]

    gamma = 0.4
    modes = [(1, 0), (0, 1)]
    eta = jnp.ones((2,))
    _, _, accept, alpha = run_tgi_rwm(
        jax.random.PRNGKey(5),
        jnp.zeros((2,)),
        log_w_fn,
        f_fn,
        gamma,
        300,
        modes,
        eta,
    )
    assert jnp.allclose(alpha, 1.0)
    assert bool(jnp.all(accept))
    assert float(jnp.mean(accept.astype(jnp.float64))) == 1.0


def test_acceptance_is_min_one_weight_ratio():
    log_w_z = jnp.array(0.2)
    log_w_prop = jnp.array(-0.3)
    alpha, _ = acceptance_probability(log_w_z, log_w_prop)
    assert jnp.allclose(alpha, jnp.exp(-0.5))
    alpha_up, _ = acceptance_probability(log_w_z, jnp.array(1.0))
    assert jnp.allclose(alpha_up, 1.0)


def test_tgi_rwm_rejects_some_when_weight_varies():
    def log_w_fn(z):
        return -0.5 * jnp.sum(z**2)

    def f_fn(z):
        return z[0]

    _, _, accept, alpha = run_tgi_rwm(
        jax.random.PRNGKey(9),
        jnp.zeros((2,)),
        log_w_fn,
        f_fn,
        0.4,
        800,
        [(1, 0)],
        jnp.array([1.0]),
    )
    rate = float(jnp.mean(accept.astype(jnp.float64)))
    assert 0.02 < rate < 0.98
    assert float(jnp.min(alpha)) < 1.0
    assert float(jnp.max(alpha)) == 1.0


def test_poisson_eta_and_all_accept_identity():
    beta = 0.6
    modes = [(1, 0), (0, 1), (2, 0)]
    c = jnp.array([0.3, -0.7, 0.4])
    mean = 1.25
    lam = mode_eigenvalues(modes, beta)
    eta = poisson_eta(c, lam)
    assert jnp.allclose(eta, c / (1.0 - lam))
    z = jax.random.normal(jax.random.PRNGKey(11), (128, 2))
    xi = jax.random.normal(jax.random.PRNGKey(12), (128, 2))
    z_prop, _ = gi_propose(z, xi, 1.0 - beta)
    phi = evaluate_modes(modes, z)
    f = mean + phi @ c
    g = g_value(z, modes, eta)
    qg = qg_value(z, modes, eta, lam)
    alpha = jnp.ones((128,))
    s = summand(f, z, z_prop, modes, eta, lam, alpha)
    assert THETA == (1.0, -1.0)
    assert jnp.allclose(s, f + qg - g)
    assert jnp.allclose(s, mean, atol=1e-10)
    assert float(jnp.var(s)) < 1e-18
    s_wrong = summand(f, z, z_prop, modes, eta, lam, alpha, theta=(-1.0, -1.0))
    assert float(jnp.max(jnp.abs(s_wrong - mean))) > 1e-2


def test_prop_51_sample_variance_vanishes():
    beta = 0.55
    modes = [(1, 0), (0, 2)]
    c = jnp.array([1.1, -0.45])
    mean = -0.2
    lam = mode_eigenvalues(modes, beta)
    eta = poisson_eta(c, lam)
    z = jax.random.normal(jax.random.PRNGKey(13), (1000, 2))
    z_prop = beta * z
    f = mean + evaluate_modes(modes, z) @ c
    s = summand(f, z, z_prop, modes, eta, lam, alpha=1.0)
    assert jnp.allclose(s, mean)
    assert float(jnp.var(s)) == pytest.approx(0.0, abs=1e-16)


def test_snis_hand_values_and_polynomial_recovery():
    w = jnp.array([1.0, 2.0])
    f_vals = jnp.array([0.5, 1.5])
    phi = jnp.array([[0.2], [0.4]])
    a = jnp.array([0.3])
    b = jnp.array([0.1])
    est = snis(w, f_vals, phi, a, b)
    numer = (1.0 * 0.5 - 0.3 * 0.2) + (2.0 * 1.5 - 0.3 * 0.4)
    denom = (1.0 - 0.1 * 0.2) + (2.0 - 0.1 * 0.4)
    assert jnp.allclose(est, numer / denom)

    z = jnp.array([[0.2, -0.4], [1.0, 0.3], [-0.7, 0.8], [0.0, 0.0]])
    modes = [(1, 0), (0, 1), (2, 0)]
    a_poly = jnp.array([0.4, -0.2, 0.5])
    mean = 1.7
    phi_poly = evaluate_modes(modes, z)
    poly = (
        mean
        + 0.4 * z[:, 0]
        - 0.2 * z[:, 1]
        + 0.5 * (z[:, 0] ** 2 - 1.0) / jnp.sqrt(2.0)
    )
    f_poly = mean + phi_poly @ a_poly
    assert jnp.allclose(f_poly, poly)
    w_one = jnp.ones((4,))
    recovered = snis(w_one, f_poly, phi_poly, a_poly, jnp.zeros((3,)))
    assert jnp.allclose(recovered, mean)


def test_pruning_threshold_is_strict():
    # Binary-exact values so the strict cut is not a float artifact.
    c_hat = jnp.array([1.0, 0.5, 0.0, 2.0])
    s_hat = jnp.array([1.0, 1.0, 1.0, 2.0])
    tau = 1.0
    n0 = 4.0
    mask = prune_mask(c_hat, s_hat, tau, n0)
    threshold = tau * s_hat**2 / n0
    expect = c_hat**2 > threshold
    assert jnp.array_equal(mask, expect)
    # 1 > 0.25, 0.25 is not greater than 0.25, 0 is not, 4 > 1.
    assert bool(mask[0]) and not bool(mask[1]) and not bool(mask[2]) and bool(mask[3])


def test_tail_penalty_and_radius():
    log_w_zeta = jnp.array([0.0, 2.0, 5.0])
    pen = tail_penalty(log_w_zeta, mean_log_w=1.0, c=1.0)
    excess = jnp.maximum(0.0, log_w_zeta - 1.0 - 1.0)
    assert jnp.allclose(pen, jnp.mean(excess**2))
    assert jnp.allclose(pen, 3.0)

    d = 2
    zeta = sample_tail(jax.random.PRNGKey(2), d, 2000)
    radius = jnp.linalg.norm(zeta, axis=-1)
    lo = jnp.sqrt(d) + 1.0
    hi = jnp.sqrt(d) + 5.0
    assert jnp.all(radius >= lo - 1e-8)
    assert jnp.all(radius < hi + 1e-8)
    direction = zeta / radius[:, None]
    assert jnp.allclose(jnp.linalg.norm(direction, axis=-1), 1.0)


def test_objective_is_negative_mean_log_w_plus_penalty():
    raw = jnp.array([0.0, 0.1, 0.0])
    params = {"raw_tril": raw, "b": jnp.array([0.2, -0.1])}

    def log_pi(x):
        return -0.5 * jnp.sum((x - jnp.array([0.2, -0.1])) ** 2)

    z = jnp.array([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.2]])
    zeta = jnp.array([[3.0, 0.0], [0.0, 3.5]])
    tail_weight = 0.05
    c = 0.5
    got = objective(params, z, zeta, log_pi, tail_weight, c)
    lw = log_w_affine(z, params, log_pi)
    lw_zeta = log_w_affine(zeta, params, log_pi)
    manual = -jnp.mean(lw) + tail_weight * tail_penalty(lw_zeta, jnp.mean(lw), c)
    assert jnp.allclose(got, manual)
