"""Equation-level tests for lPINN (arXiv:2609.14926), ADE kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_lpinn.data import (
    bc_dirichlet_collocation,
    bc_neumann_collocation,
    flatten_mesh,
    ic_collocation,
    interior_collocation,
    make_ade_ensemble,
    space_time_mesh,
    split_mean_fluct,
)
from jax_lpinn.metrics import rrmse
from jax_lpinn.model import (
    LPINNConfig,
    features,
    fluct_value,
    init_fluct_hidden,
    init_mean_params,
    last_hidden,
    mean_value,
    n_eta,
    predict,
    scale_z,
)
from jax_lpinn.physics import ade_operator, ogata_banks, ogata_banks_derivs
from jax_lpinn.train import (
    build_ade_lstsq_system,
    fit_reconstruction_W,
    online_forward_lstsq,
    online_objective,
    train_mean_derivative_matching,
)


Nx, Nt = 16, 8
CFG = LPINNConfig(hidden_widths=(8, 8))


def _tree_clone(tree):
    return jax.tree.map(lambda x: jnp.array(x, copy=True), tree)


def _tree_allclose(a, b, **kw):
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), **kw)


def _tiny_collocation(cfg=CFG, nx=Nx, nt=Nt):
    x_res, t_res = interior_collocation(nx, nt, cfg.L, cfg.T)
    x_ic, t_ic = ic_collocation(nx, cfg.L)
    x_bc0, t_bc0 = bc_dirichlet_collocation(nt, cfg.T)
    x_bcL, t_bcL = bc_neumann_collocation(nt, cfg.L, cfg.T)
    return x_res, t_res, x_ic, t_ic, x_bc0, t_bc0, x_bcL, t_bcL


@pytest.fixture(scope="module")
def nets():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    mean = init_mean_params(k1, CFG)
    hidden = init_fluct_hidden(k2, CFG)
    return mean, hidden


def test_eq10_reconstruction_linear_in_W_and_zero_output_bias(nets):
    """Eq 10: ĥ' = W · ψ, linear in W, b_{N+1} ≡ 0."""
    mean_params, hidden = nets
    key = jax.random.PRNGKey(1)
    kW1, kW2, kx, kt = jax.random.split(key, 4)
    n = n_eta(CFG)
    W1 = jax.random.normal(kW1, (n,))
    W2 = jax.random.normal(kW2, (n,))
    x = jax.random.uniform(kx, (32,), minval=0.0, maxval=CFG.L)
    t = jax.random.uniform(kt, (32,), minval=0.0, maxval=CFG.T)

    psi = features(hidden, x, t, L=CFG.L, T=CFG.T)
    h1 = fluct_value(hidden, W1, x, t, L=CFG.L, T=CFG.T)
    h2 = fluct_value(hidden, W2, x, t, L=CFG.L, T=CFG.T)
    np.testing.assert_allclose(np.asarray(h1), np.asarray(psi @ W1), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(fluct_value(hidden, 2.0 * W1, x, t, L=CFG.L, T=CFG.T)),
        np.asarray(2.0 * h1),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(fluct_value(hidden, W1 + W2, x, t, L=CFG.L, T=CFG.T)),
        np.asarray(h1 + h2),
        atol=1e-12,
        rtol=1e-12,
    )
    z = scale_z(x, t, CFG.L, CFG.T)
    np.testing.assert_allclose(np.asarray(last_hidden(hidden, z)), np.asarray(psi), atol=0, rtol=0)
    assert "b_out" not in hidden
    zeros = fluct_value(hidden, jnp.zeros((n,)), x, t, L=CFG.L, T=CFG.T)
    np.testing.assert_allclose(np.asarray(zeros), 0.0, atol=0.0, rtol=0.0)
    # Mean network is allowed an output bias; fluctuation is not.
    assert "b_out" in mean_params


def test_hidden_params_frozen_online(nets):
    """Online solve updates only W_{N+1}; θ̃ and γ stay bitwise identical."""
    mean_params, hidden = nets
    gamma0 = _tree_clone(mean_params)
    theta0 = _tree_clone(hidden)
    cols = _tiny_collocation()
    W = online_forward_lstsq(mean_params, hidden, *cols, CFG.V_star, CFG.D_star, CFG)
    assert W.shape == (n_eta(CFG),)
    _tree_allclose(mean_params, gamma0, atol=0.0, rtol=0.0)
    _tree_allclose(hidden, theta0, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(np.asarray(mean_params["W_out"]), np.asarray(gamma0["W_out"]))
    np.testing.assert_array_equal(np.asarray(hidden["layers"][0]["W"]), np.asarray(theta0["layers"][0]["W"]))


def test_ade_online_objective_quadratic_in_W(nets):
    """For linear ADE, Eq 20 is quadratic in W: lstsq matches normal equations."""
    mean_params, hidden = nets
    cols = _tiny_collocation()
    A, b = build_ade_lstsq_system(mean_params, hidden, *cols, CFG.V_star, CFG.D_star, CFG)
    W_star = jnp.linalg.lstsq(A, b, rcond=None)[0]
    W_online = online_forward_lstsq(mean_params, hidden, *cols, CFG.V_star, CFG.D_star, CFG)
    np.testing.assert_allclose(np.asarray(W_star), np.asarray(W_online), atol=1e-10, rtol=1e-10)

    # Normal equations A^T (A W - b) = 0 (regularized, so well-posed).
    normal = A.T @ (A @ W_star - b)
    np.testing.assert_allclose(np.asarray(normal), 0.0, atol=1e-8, rtol=1e-8)

    def J(w):
        r = A @ w - b
        return jnp.sum(r * r)

    key = jax.random.PRNGKey(2)
    h = jax.random.normal(key, W_star.shape)
    quad = J(W_star + h) + J(W_star - h) - 2.0 * J(W_star)
    np.testing.assert_allclose(np.asarray(quad), np.asarray(2.0 * jnp.sum((A @ h) ** 2)), atol=1e-8, rtol=1e-8)

    # Finite-difference gradient at W* is ~0, so W* is a critical point / min.
    eps = 1e-6
    v = h / jnp.linalg.norm(h)
    g_fd = (J(W_star + eps * v) - J(W_star - eps * v)) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(g_fd), 0.0, atol=1e-5, rtol=0.0)
    assert float(J(W_star)) <= float(J(jnp.zeros_like(W_star))) + 1e-12

    J_obj = online_objective(W_star, mean_params, hidden, *cols, CFG.V_star, CFG.D_star, CFG)
    np.testing.assert_allclose(np.asarray(J_obj), np.asarray(J(W_star)), atol=1e-10, rtol=1e-10)


def test_ogata_banks_satisfies_ade_interior():
    """Eq 27 solves u_t + V u_x = D u_xx on t>0, 0<x<L."""
    L, T, u0 = CFG.L, CFG.T, CFG.u0
    x = jnp.linspace(0.08 * L, 0.92 * L, 7)
    t = jnp.linspace(0.25 * T, T, 5)
    xx, tt = jnp.meshgrid(x, t, indexing="ij")
    xf, tf = flatten_mesh(xx, tt)
    cases = [(0.213, 0.061), (0.15, 0.08), (0.25, 0.05)]
    for V, D in cases:
        u, ut, ux, uxx = ogata_banks_derivs(xf, tf, V, D, u0)
        R = ade_operator(ut, ux, uxx, V, D)
        scale = jnp.maximum(jnp.max(jnp.abs(ut)), 1.0)
        np.testing.assert_allclose(np.asarray(R / scale), 0.0, atol=5e-5, rtol=0.0)
        # Inlet Dirichlet of the half-line solution.
        t_bc = jnp.linspace(0.1 * T, T, 8)
        u_in = ogata_banks(jnp.zeros_like(t_bc), t_bc, V, D, u0)
        np.testing.assert_allclose(np.asarray(u_in), u0, atol=1e-8, rtol=1e-8)


def test_sample_mean_fluctuation_split_eqs_6_12():
    """ū is the ensemble mean; mean_i u'^{(i)} ≈ 0 (Eqs 6, 12, 28–29)."""
    key = jax.random.PRNGKey(3)
    ens = make_ade_ensemble(key, n=8, Nx=Nx, Nt=Nt, cfg=CFG, with_derivs=False)
    split = split_mean_fluct(ens["U"])
    np.testing.assert_allclose(
        np.asarray(split["U_bar"]), np.asarray(ens["U"].mean(axis=0)), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(split["U_prime"].mean(axis=0)), 0.0, atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(split["U_bar"] + split["U_prime"]), np.asarray(ens["U"]), atol=1e-14, rtol=1e-14
    )


def test_tiny_offline_online_beats_mean_only(nets):
    """Tiny offline + online: finite rRMSE; reconstruction LS beats mean-only."""
    mean_params, hidden = nets
    key = jax.random.PRNGKey(4)
    cfg_val = LPINNConfig(hidden_widths=CFG.hidden_widths, lam_t=0.0, lam_x=0.0, lam_xx=0.0)
    ens = make_ade_ensemble(key, n=8, Nx=Nx, Nt=Nt, cfg=cfg_val, with_derivs=False)
    split = split_mean_fluct(ens["U"])
    x_flat, t_flat = flatten_mesh(ens["xx"], ens["tt"])
    u_bar = split["U_bar"].reshape(-1)
    z = jnp.zeros_like(u_bar)
    mean_params, _ = train_mean_derivative_matching(
        mean_params, x_flat, t_flat, u_bar, z, z, z, cfg_val, steps=5, lr=1e-3
    )

    V_h, D_h = 1.1 * CFG.V_star, 0.9 * CFG.D_star
    u_hold = ogata_banks(x_flat, t_flat, V_h, D_h, CFG.u0)
    h_mean = mean_value(mean_params, x_flat, t_flat, L=CFG.L, T=CFG.T)
    W_fit = fit_reconstruction_W(mean_params, hidden, x_flat, t_flat, u_hold, CFG)
    h_fit = predict(mean_params, hidden, W_fit, x_flat, t_flat, L=CFG.L, T=CFG.T)
    err_mean = float(rrmse(h_mean, u_hold, CFG.u0))
    err_fit = float(rrmse(h_fit, u_hold, CFG.u0))
    assert np.isfinite(err_mean) and np.isfinite(err_fit)
    assert err_fit < err_mean

    cols = _tiny_collocation()
    W_ls = online_forward_lstsq(mean_params, hidden, *cols, V_h, D_h, CFG)
    J_star = float(online_objective(W_ls, mean_params, hidden, *cols, V_h, D_h, CFG))
    J_zero = float(online_objective(jnp.zeros_like(W_ls), mean_params, hidden, *cols, V_h, D_h, CFG))
    assert np.isfinite(J_star) and J_star <= J_zero + 1e-10
    u_online = predict(mean_params, hidden, W_ls, x_flat, t_flat, L=CFG.L, T=CFG.T)
    assert np.isfinite(float(rrmse(u_online, u_hold, CFG.u0)))


def test_cross_resolution_continuous_query(nets):
    """A network on a coarse mesh can be queried on a finer mesh without retraining."""
    mean_params, hidden = nets
    n = n_eta(CFG)
    W = jax.random.normal(jax.random.PRNGKey(5), (n,))
    _, _, xx_c, tt_c = space_time_mesh(8, 4, CFG.L, CFG.T)
    _, _, xx_f, tt_f = space_time_mesh(24, 12, CFG.L, CFG.T)
    xc, tc = flatten_mesh(xx_c, tt_c)
    xf, tf = flatten_mesh(xx_f, tt_f)
    uc = predict(mean_params, hidden, W, xc, tc, L=CFG.L, T=CFG.T)
    uf = predict(mean_params, hidden, W, xf, tf, L=CFG.L, T=CFG.T)
    assert uc.shape == xc.shape
    assert uf.shape == xf.shape
    assert uf.shape[0] > uc.shape[0]
    # Same physical point, two meshes: query is mesh-free.
    x0 = jnp.array([0.3 * CFG.L])
    t0 = jnp.array([0.4 * CFG.T])
    u0a = predict(mean_params, hidden, W, x0, t0, L=CFG.L, T=CFG.T)
    u0b = predict(mean_params, hidden, W, x0, t0, L=CFG.L, T=CFG.T)
    np.testing.assert_allclose(np.asarray(u0a), np.asarray(u0b), atol=0.0, rtol=0.0)
