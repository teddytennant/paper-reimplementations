"""Properties from arXiv:2609.03306 section 3, on low-dimensional toys."""

import jax
import jax.numpy as jnp
import numpy as np

from adaptive_spectral_bandwidth.bandwidth import (
    bandwidth_grid,
    dim_mst,
    effective_rank,
    gaussian_kernel,
    loglog_slope,
    select_bandwidth,
    sharpen_weights,
    spectral_criterion,
)

jax.config.update("jax_enable_x64", True)


def test_effective_rank_identity_and_ones():
    n = 8
    r_i = float(effective_rank(jnp.eye(n)))
    r_o = float(effective_rank(jnp.ones((n, n))))
    np.testing.assert_allclose(r_i, n, atol=1e-10)
    np.testing.assert_allclose(r_o, 1.0, atol=1e-10)


def test_reff_limits_on_a_line():
    """Tiny sigma: r_eff ~ n. Huge sigma: r_eff ~ 1."""
    t = jnp.linspace(0.0, 1.0, 16)
    x = jnp.stack([t, jnp.zeros_like(t)], axis=1)
    r_small = float(effective_rank(gaussian_kernel(x, 1e-4)))
    r_large = float(effective_rank(gaussian_kernel(x, 1e2)))
    assert r_small > 12.0
    assert r_large < 1.5


def test_dimmst_line_vs_plane():
    """1-d line recovers near 1; 2-d subspace of R^8 recovers near 2, not 8."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    t = jax.random.normal(k1, (40, 1))
    line = jnp.concatenate([t, jnp.zeros((40, 7))], axis=1)
    d1 = float(dim_mst(line, k1, n_sizes=5, n_rep=5))
    assert d1 < 1.7, f"1-d estimate {d1}"

    plane = jax.random.normal(k2, (40, 2))
    pad = jnp.zeros((40, 6))
    x2 = jnp.concatenate([plane, pad], axis=1)
    d2 = float(dim_mst(x2, k2, n_sizes=5, n_rep=5))
    assert 1.3 <= d2 <= 3.5, f"2-d estimate {d2}"


def test_j_minimizer_not_at_grid_ends():
    """On a line, sigma_star sits between the identity and rank-one limits."""
    t = jnp.linspace(-1.0, 1.0, 24)
    x = jnp.stack([t, 0.05 * jax.random.normal(jax.random.key(1), (24,))], axis=1)
    star, sigmas, J = select_bandwidth(x, jax.random.key(2), k_mle=5, n_grid=12)
    i = int(jnp.argmin(J))
    assert 0 < i < len(sigmas) - 1, f"minimizer at endpoint i={i}, sigmas={sigmas}, J={J}"
    assert float(star) > float(sigmas[0])
    assert float(star) < float(sigmas[-1])


def test_reff_crossing_near_dhat():
    """The minimizer of |r_eff - d| / d is near the r_eff ≈ d crossing, not a limit.

    d_hat sits between the identity limit (r_eff ~ n) and the rank-one limit
    (r_eff ~ 1), matching Fig. 1 of the paper.
    """
    t = jnp.linspace(-1.0, 1.0, 20)
    x = jnp.stack([t, jnp.zeros_like(t)], axis=1)
    d_hat = 4.0
    sigmas = jnp.logspace(-3, 1, 16)
    reffs = np.array([float(effective_rank(gaussian_kernel(x, s))) for s in sigmas])
    term1 = np.abs(reffs - d_hat) / d_hat
    i = int(np.argmin(term1))
    assert 0 < i < len(sigmas) - 1
    # Identity and rank-one ends are worse than the crossing.
    assert term1[0] > 0.5
    assert term1[-1] > 0.5
    assert term1[i] < term1[0]
    assert term1[i] < term1[-1]


def test_loglog_slope_peak_on_circle():
    """ell(sigma) is not monotone with both peaks at the grid endpoints only."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, 30, endpoint=False)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    sigmas = jnp.logspace(-2, 1, 16)
    ell = np.array(loglog_slope(x, sigmas))
    i = int(np.argmax(ell))
    # Peak of the scaling slope sits in the interior of a wide grid.
    assert 0 < i < len(sigmas) - 1, f"peak at endpoint i={i}, ell={ell}"


def test_sigma_matching_reff_to_dhat_is_finite():
    """Isotropic 2-d Gaussian: a finite positive sigma has r_eff ≈ d_hat."""
    x = jax.random.normal(jax.random.key(3), (30, 2))
    d_hat = float(dim_mst(x, jax.random.key(4)))
    sigmas = jnp.logspace(-2, 1, 20)
    reffs = np.array([float(effective_rank(gaussian_kernel(x, s))) for s in sigmas])
    i = int(np.argmin(np.abs(reffs - d_hat)))
    assert np.isfinite(float(sigmas[i]))
    assert float(sigmas[i]) > 0
    assert 0 < i < len(sigmas) - 1


def test_sharpen_weights_concentrate_and_flatten():
    w = jnp.array([0.1, 0.2, 0.7])
    # p > 1 when sigma_star < sigma_g (dense region).
    w_hi = np.array(sharpen_weights(w, sigma_star=jnp.array(0.5), sigma_g=jnp.array(1.0)))
    w_lo = np.array(sharpen_weights(w, sigma_star=jnp.array(2.0), sigma_g=jnp.array(1.0)))
    np.testing.assert_allclose(w_hi.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(w_lo.sum(), 1.0, atol=1e-12)
    assert np.all(w_hi >= 0) and np.all(w_lo >= 0)
    # Concentration: the largest weight grows when p > 1.
    assert w_hi[2] > 0.7
    # Flattening: the largest weight shrinks when p < 1.
    assert w_lo[2] < 0.7
    # Clip bounds 0.2 and 2.0: extreme ratios still produce finite weights.
    w_clip_hi = np.array(sharpen_weights(w, sigma_star=jnp.array(1e-6), sigma_g=jnp.array(1.0)))
    w_clip_lo = np.array(sharpen_weights(w, sigma_star=jnp.array(1e6), sigma_g=jnp.array(1.0)))
    assert np.isfinite(w_clip_hi).all() and np.isfinite(w_clip_lo).all()


def test_grid_logspaced_twelve():
    g = bandwidth_grid(1.0, 2.0, n_grid=12)
    assert g.shape == (12,)
    np.testing.assert_allclose(float(g[0]), 0.05, rtol=1e-6)
    np.testing.assert_allclose(float(g[-1]), 6.0, rtol=1e-6)
