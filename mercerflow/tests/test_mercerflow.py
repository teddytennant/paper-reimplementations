"""Equation-level checks for MercerFlow. Small grids, CPU, no GluonTS."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
import pytest

from mercerflow import (
    apply_velocity_mlp,
    build_basis,
    conditional_moments,
    count_parameters,
    dct_basis,
    decode,
    encode,
    euler_forecast,
    euler_integrate,
    flow_matching_loss,
    fourier_basis,
    gram_matrix,
    init_velocity_mlp,
    interpolant,
    jacobi_condition_number,
    jacobi_scaled,
    mercer_basis,
    ou_kernel,
    partition_gram,
    pca_basis,
    prior_coefficient_covariance,
    raw_basis,
    sample_prior,
    shared_context_velocity,
    sinusoidal_time_embedding,
    train_step,
    training_step,
)

DTYPE = jnp.float64


def test_ou_kernel_and_gram_match_equation_1():
    length = 5
    ell = 2.0
    eps = 0.01
    index = jnp.arange(length, dtype=DTYPE)
    kernel = ou_kernel(index[:, None], index[None, :], ell)
    for i in range(length):
        for j in range(length):
            expected = math.exp(-abs(i - j) / ell)
            assert abs(float(kernel[i, j]) - expected) < 1e-12
    K = gram_matrix(length, ell, eps, dtype=DTYPE)
    for i in range(length):
        for j in range(length):
            expected = math.exp(-abs(i - j) / ell)
            if i == j:
                expected += eps
            assert abs(float(K[i, j]) - expected) < 1e-12


def test_partition_reconstructs_gram_equation_2():
    K = gram_matrix(6, 1.5, 1e-3, dtype=DTYPE)
    n_past = 2
    Kpp, Kpf, Kfp, Kff = partition_gram(K, n_past)
    assert Kpp.shape == (2, 2)
    assert Kpf.shape == (2, 4)
    assert Kfp.shape == (4, 2)
    assert Kff.shape == (4, 4)
    top = jnp.concatenate([Kpp, Kpf], axis=1)
    bottom = jnp.concatenate([Kfp, Kff], axis=1)
    assert jnp.allclose(jnp.concatenate([top, bottom], axis=0), K)


def test_conditional_moments_match_equation_3_known_case():
    # L = 2, Lp = 1, ell = 1, no jitter. Closed form of equation (3).
    K = gram_matrix(2, 1.0, 0.0, dtype=DTYPE)
    y_past = jnp.array([2.0], dtype=DTYPE)
    mu, sigma = conditional_moments(K, y_past, 1)
    decay = math.exp(-1.0)
    assert abs(float(mu[0]) - 2.0 * decay) < 1e-12
    assert abs(float(sigma[0, 0]) - (1.0 - decay ** 2)) < 1e-12


def test_conditional_moments_match_explicit_inverse():
    K = gram_matrix(6, 1.4, 1e-2, dtype=DTYPE)
    y_past = jnp.array([0.2, -0.4, 1.1], dtype=DTYPE)
    n_past = 3
    mu, sigma = conditional_moments(K, y_past, n_past)
    Kpp, Kpf, Kfp, Kff = partition_gram(K, n_past)
    inv_past = jnp.linalg.inv(Kpp)
    mu_ref = Kfp @ inv_past @ y_past
    sigma_ref = Kff - Kfp @ inv_past @ Kpf
    sigma_ref = 0.5 * (sigma_ref + sigma_ref.T)
    assert jnp.allclose(mu, mu_ref, atol=1e-10)
    assert jnp.allclose(sigma, sigma_ref, atol=1e-10)


def test_conditional_batch_matches_each_row():
    K = gram_matrix(5, 1.2, 1e-3, dtype=DTYPE)
    y_past = jnp.array(
        [[0.2, -0.5, 1.0], [1.5, 0.0, -0.3]],
        dtype=DTYPE,
    )
    mu, sigma = conditional_moments(K, y_past, 3)
    for row in range(2):
        mu_row, sigma_row = conditional_moments(K, y_past[row], 3)
        assert jnp.allclose(mu[row], mu_row)
        assert jnp.allclose(sigma, sigma_row)


@pytest.mark.parametrize("name", ["raw", "fourier", "dct", "mercer", "pca"])
def test_codecs_are_orthonormal(name):
    length = 8
    K = gram_matrix(length, 2.5, 1e-3, dtype=DTYPE)
    windows = sample_prior(jax.random.PRNGKey(0), K, n=12) + 1.5
    Phi = build_basis(name, length, K=K, windows=windows, dtype=DTYPE)
    eye = jnp.eye(length, dtype=DTYPE)
    assert Phi.shape == (length, length)
    assert jnp.allclose(Phi.T @ Phi, eye, atol=1e-8)
    assert jnp.allclose(Phi @ Phi.T, eye, atol=1e-8)


def test_mercer_eigendecomposition_covariance_and_signs():
    K = gram_matrix(12, 3.0, 1e-4, dtype=DTYPE)
    Phi, eigenvalues = mercer_basis(K)
    assert jnp.all(eigenvalues[:-1] >= eigenvalues[1:] - 1e-12)
    assert jnp.all(Phi[0] >= 0.0)
    reconstructed = Phi @ jnp.diag(eigenvalues) @ Phi.T
    assert jnp.allclose(reconstructed, K, atol=1e-8)
    # Exact identity: Cov(Phi.T y) = Lambda for y ~ N(0, K).
    coefficient_cov = Phi.T @ K @ Phi
    assert jnp.allclose(coefficient_cov, jnp.diag(eigenvalues), atol=1e-8)


def test_fourier_matches_hand_built_columns():
    length = 8
    Phi = fourier_basis(length, dtype=DTYPE)
    s = jnp.arange(length, dtype=DTYPE)
    dc = jnp.full((length,), length ** -0.5, dtype=DTYPE)
    assert jnp.allclose(Phi[:, 0], dc, atol=1e-12)
    harmonic = 1
    cosine = math.sqrt(2.0 / length) * jnp.cos(2.0 * math.pi * harmonic * s / length)
    sine = math.sqrt(2.0 / length) * jnp.sin(2.0 * math.pi * harmonic * s / length)
    assert jnp.allclose(Phi[:, 1], cosine, atol=1e-12)
    assert jnp.allclose(Phi[:, 2], sine, atol=1e-12)
    nyquist = (length ** -0.5) * jnp.cos(math.pi * s)
    assert jnp.allclose(Phi[:, -1], nyquist, atol=1e-12)
    # Odd length has no Nyquist column. The last pair is the highest harmonic.
    odd = 7
    Phi_odd = fourier_basis(odd, dtype=DTYPE)
    s_odd = jnp.arange(odd, dtype=DTYPE)
    highest = (odd - 1) // 2
    last_sine = math.sqrt(2.0 / odd) * jnp.sin(2.0 * math.pi * highest * s_odd / odd)
    assert jnp.allclose(Phi_odd[:, -1], last_sine, atol=1e-12)


def test_dct_matches_hand_built_column():
    length = 7
    Phi = dct_basis(length, dtype=DTYPE)
    s = jnp.arange(length, dtype=DTYPE)
    dc = jnp.full((length,), length ** -0.5, dtype=DTYPE)
    assert jnp.allclose(Phi[:, 0], dc, atol=1e-12)
    mode = 3
    expected = math.sqrt(2.0 / length) * jnp.cos(
        math.pi * mode * (s + 0.5) / length
    )
    assert jnp.allclose(Phi[:, mode], expected, atol=1e-12)


def test_pca_encode_does_not_subtract_mean_and_matches_equation_8():
    K = gram_matrix(8, 2.0, 1e-3, dtype=DTYPE)
    windows = sample_prior(jax.random.PRNGKey(1), K, n=20) + 3.0
    Phi, eigenvalues = pca_basis(windows)
    assert jnp.all(Phi[0] >= 0.0)
    assert jnp.all(eigenvalues[:-1] >= eigenvalues[1:] - 1e-12)
    mean = jnp.mean(windows, axis=0)
    centered = windows - mean
    covariance = (centered.T @ centered) / windows.shape[0]
    assert jnp.allclose(Phi.T @ covariance @ Phi, jnp.diag(eigenvalues), atol=1e-8)
    y = windows[0]
    coefficients = encode(Phi, y)
    assert jnp.allclose(coefficients, Phi.T @ y, atol=1e-12)
    centered_codes = Phi.T @ (y - mean)
    assert not jnp.allclose(coefficients, centered_codes, atol=1e-5)
    assert jnp.allclose(decode(Phi, coefficients), y, atol=1e-8)


def test_flow_interpolant_and_isometry():
    length = 10
    n_past = 4
    K = gram_matrix(length, 1.7, 1e-3, dtype=DTYPE)
    windows = sample_prior(jax.random.PRNGKey(2), K, n=6)
    y_past = windows[0, :n_past]
    y0 = windows[1, n_past:]
    y_future = windows[2, n_past:]
    for name in ("raw", "fourier", "dct", "mercer", "pca"):
        Phi = build_basis(name, length, K=K, windows=windows, dtype=DTYPE)
        source = jnp.concatenate([y_past, y0])
        target = jnp.concatenate([y_past, y_future])
        c0 = encode(Phi, source)
        c1 = encode(Phi, target)
        t = jnp.asarray(0.3, dtype=DTYPE)
        ct, u = interpolant(c0, c1, t)
        assert jnp.allclose(ct, (1.0 - t) * c0 + t * c1, atol=1e-12)
        assert jnp.allclose(u, c1 - c0, atol=1e-12)
        assert jnp.allclose(interpolant(c0, c1, 0.0)[0], c0, atol=1e-12)
        assert jnp.allclose(interpolant(c0, c1, 1.0)[0], c1, atol=1e-12)
        u_shared = shared_context_velocity(Phi, y0, y_future)
        assert jnp.allclose(u, u_shared, atol=1e-8)
        assert jnp.allclose(
            jnp.linalg.norm(u),
            jnp.linalg.norm(y_future - y0),
            atol=1e-8,
        )


def test_mlp_matches_equation_27_including_zero_init():
    length, hidden, embed = 6, 4, 4
    params = init_velocity_mlp(
        jax.random.PRNGKey(1),
        length,
        hidden=hidden,
        embed_dim=embed,
        dtype=DTYPE,
    )
    assert int(params["W1"].shape[0]) == hidden
    assert jnp.all(params["W_t"] == 0.0)
    assert jnp.all(params["b_t"] == 0.0)
    c = jnp.linspace(-1.0, 1.0, length, dtype=DTYPE)
    t_early = jnp.asarray(0.2, dtype=DTYPE)
    t_late = jnp.asarray(0.8, dtype=DTYPE)
    v_early = apply_velocity_mlp(params, t_early, c)
    v_late = apply_velocity_mlp(params, t_late, c)
    h = params["W1"] @ c + params["b1"]
    v_ref = params["W2"] @ jax.nn.gelu(h, approximate=False) + params["b2"]
    assert jnp.allclose(v_early, v_ref, atol=1e-12)
    # Zero-initialised gamma and beta ignore flow time.
    assert jnp.allclose(v_early, v_late, atol=1e-12)

    modulated = dict(params)
    modulated["W_t"] = 0.01 * jnp.arange(
        2 * hidden * embed, dtype=DTYPE
    ).reshape(2 * hidden, embed)
    modulated["b_t"] = jnp.linspace(-0.2, 0.2, 2 * hidden, dtype=DTYPE)
    t = jnp.asarray(0.37, dtype=DTYPE)
    v = apply_velocity_mlp(modulated, t, c)
    e_t = sinusoidal_time_embedding(t, embed)
    gamma_beta = modulated["W_t"] @ e_t + modulated["b_t"]
    gamma, beta = gamma_beta[:hidden], gamma_beta[hidden:]
    h = modulated["W1"] @ c + modulated["b1"]
    h_tilde = jax.nn.gelu((1.0 + gamma) * h + beta, approximate=False)
    v_equation = modulated["W2"] @ h_tilde + modulated["b2"]
    assert jnp.allclose(v, v_equation, atol=1e-12)
    assert not jnp.allclose(v, v_ref, atol=1e-6)


def test_hidden_width_default_and_l384_parameter_count():
    import inspect

    assert inspect.signature(init_velocity_mlp).parameters["hidden"].default == 64
    assert inspect.signature(euler_integrate).parameters["n_steps"].default == 20
    assert inspect.signature(euler_forecast).parameters["n_steps"].default == 20
    params = init_velocity_mlp(jax.random.PRNGKey(0), length=384, hidden=64)
    embed = 64
    expected = (
        64 * 384
        + 64
        + 384 * 64
        + 384
        + (2 * 64) * embed
        + (2 * 64)
    )
    assert count_parameters(params) == expected


def test_one_training_step_reduces_fixed_batch_loss():
    length = 8
    n_past = 3
    K = gram_matrix(length, 1.5, 1e-3, dtype=DTYPE)
    Phi, _ = mercer_basis(K)
    windows = sample_prior(jax.random.PRNGKey(4), K, n=6)
    y_past = windows[:, :n_past]
    y_future = windows[:, n_past:]
    y0 = sample_prior(jax.random.PRNGKey(5), K, n=6)[:, n_past:]
    c0 = encode(Phi, jnp.concatenate([y_past, y0], axis=-1))
    c1 = encode(Phi, jnp.concatenate([y_past, y_future], axis=-1))
    t = jnp.linspace(0.1, 0.9, y_past.shape[0], dtype=DTYPE)
    params = init_velocity_mlp(
        jax.random.PRNGKey(6), length, hidden=8, dtype=DTYPE
    )
    optimizer = optax.sgd(1e-3)
    opt_state = optimizer.init(params)
    loss_before = flow_matching_loss(params, c0, c1, t)
    params, opt_state, loss_step = train_step(
        params, opt_state, optimizer, c0, c1, t
    )
    loss_after = flow_matching_loss(params, c0, c1, t)
    assert math.isfinite(float(loss_step))
    assert math.isfinite(float(loss_after))
    assert float(loss_after) < float(loss_before)


def test_training_step_on_synthetic_ou_is_finite():
    length = 8
    n_past = 4
    K = gram_matrix(length, 2.0, 1e-3, dtype=DTYPE)
    Phi = build_basis("fourier", length, dtype=DTYPE)
    windows = sample_prior(jax.random.PRNGKey(7), K, n=8)
    params = init_velocity_mlp(
        jax.random.PRNGKey(8), length, hidden=8, dtype=DTYPE
    )
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    params, opt_state, loss = training_step(
        params,
        opt_state,
        optimizer,
        Phi,
        K,
        windows[:, :n_past],
        windows[:, n_past:],
        jax.random.PRNGKey(9),
    )
    assert math.isfinite(float(loss))
    assert all(math.isfinite(float(leaf.sum())) for leaf in jax.tree_util.tree_leaves(params))


def test_euler_sample_shape_and_one_step_update():
    length = 8
    n_past = 5
    K = gram_matrix(length, 1.8, 1e-3, dtype=DTYPE)
    Phi, _ = mercer_basis(K)
    params = init_velocity_mlp(
        jax.random.PRNGKey(3), length, hidden=6, dtype=DTYPE
    )
    y_past = sample_prior(jax.random.PRNGKey(10), K, n=1)[0, :n_past]
    forecast = euler_forecast(
        params, Phi, y_past, K, jax.random.PRNGKey(11), n_steps=3
    )
    assert forecast.shape == (length - n_past,)
    assert jnp.all(jnp.isfinite(forecast))

    c0 = jnp.linspace(-0.5, 0.5, length, dtype=DTYPE)
    v0 = apply_velocity_mlp(params, 0.0, c0)
    assert jnp.allclose(euler_integrate(params, c0, n_steps=1), c0 + v0, atol=1e-12)
    step = euler_integrate(params, c0, n_steps=2)
    midpoint = c0 + 0.5 * v0
    v_mid = apply_velocity_mlp(params, 0.5, midpoint)
    assert jnp.allclose(step, midpoint + 0.5 * v_mid, atol=1e-12)


def test_proposition_2_jacobi_condition():
    length = 16
    eps = 1e-4
    long_ell = 30.0
    K_long = gram_matrix(length, long_ell, eps, dtype=DTYPE)
    K_short = gram_matrix(length, 0.5, eps, dtype=DTYPE)
    Phi_mercer, eigenvalues = mercer_basis(K_long)
    Phi_raw = raw_basis(length, dtype=DTYPE)
    A_mercer = prior_coefficient_covariance(Phi_mercer, K_long)
    A_raw = prior_coefficient_covariance(Phi_raw, K_long)
    # Mercer: A = Lambda, so D A D = I up to numerical jitter.
    assert jnp.allclose(A_mercer, jnp.diag(eigenvalues), atol=1e-8)
    assert jnp.allclose(jacobi_scaled(A_mercer), jnp.eye(length, dtype=DTYPE), atol=1e-6)
    kappa_mercer = float(jacobi_condition_number(A_mercer))
    kappa_raw = float(jacobi_condition_number(A_raw))
    kappa_short = float(
        jacobi_condition_number(prior_coefficient_covariance(Phi_raw, K_short))
    )
    assert abs(kappa_mercer - 1.0) < 1e-5
    assert kappa_mercer < kappa_raw
    # Raw diagonal is constant, so D is a scalar multiple of I.
    raw_diag = jnp.diag(A_raw)
    assert jnp.allclose(raw_diag, raw_diag[0], atol=1e-12)
    raw_eigs = jnp.linalg.eigvalsh(A_raw)
    assert abs(kappa_raw - float(raw_eigs[-1] / raw_eigs[0])) < 1e-6
    assert kappa_raw > kappa_short
    assert kappa_raw > 10.0
