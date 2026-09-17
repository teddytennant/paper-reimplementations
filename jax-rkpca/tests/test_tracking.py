"""Streaming Algorithm 1 on a short synthetic linear subspace."""

from __future__ import annotations

import numpy as np

from jax_rkpca.metrics import generate_synthetic_stream, grassmann_distance
from jax_rkpca.update import init_state, step, track


def _run_linear_stream(seed, rotate, n=200, D=8, d=2, eta=0.01, alpha0=0.1):
    X, Q, _ = generate_synthetic_stream(
        n=n,
        D=D,
        d=d,
        eigenvalues=[6.0, 3.0],
        noise_std=0.08,
        seed=seed,
    )
    U = np.eye(D)
    state, cs = track(
        X,
        U,
        d=d,
        eta=eta,
        alpha0=alpha0,
        kernel="linear",
        rotate=rotate,
    )
    dist = float(grassmann_distance(state.Gamma, Q))
    return dist, np.asarray(cs), np.asarray(state.Gamma)


def test_loadings_shape_on_stream():
    dist, cs, G = _run_linear_stream(seed=0, rotate=True, n=30)
    assert cs.shape == (30, 2)
    np.testing.assert_allclose(G.T @ G, np.eye(2), atol=1e-5)
    assert np.isfinite(dist)


def test_gamma_stays_orthonormal_after_many_steps():
    rng = np.random.default_rng(11)
    M, d = 6, 2
    state = init_state(M, d)
    for _ in range(40):
        h = rng.normal(size=(M,))
        state, c_t, _ = step(state, h, eta=0.05, alpha0=0.3)
        assert np.asarray(c_t).shape == (d,)
    G = np.asarray(state.Gamma)
    np.testing.assert_allclose(G.T @ G, np.eye(d), atol=1e-5)


def test_rotation_beats_gradient_only_mean_over_seeds():
    seeds = (0, 1, 2)
    full = []
    grad = []
    for seed in seeds:
        d_full, _, _ = _run_linear_stream(seed=seed, rotate=True)
        d_grad, _, _ = _run_linear_stream(seed=seed, rotate=False)
        full.append(d_full)
        grad.append(d_grad)
    mean_full = float(np.mean(full))
    mean_grad = float(np.mean(grad))
    assert np.isfinite(mean_full) and np.isfinite(mean_grad)
    assert mean_full < mean_grad, (
        f"expected rotation-enabled mean Grassmann {mean_full:.4f} "
        f"< gradient-only {mean_grad:.4f}; per-seed full={full} grad={grad}"
    )
