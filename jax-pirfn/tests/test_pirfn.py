"""CPU tests for jax-pirfn."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_pirfn.features import (
    cosine_features,
    init_product_features,
    init_random_features,
    product_features,
    sample_bias,
    sample_omega,
)
from jax_pirfn.losses import loss_terms, pirfn_loss, relative_l2
from jax_pirfn.pde import (
    apply_1d,
    apply_product,
    heat_1d_exact,
    heat_residual,
    helmholtz_1d_source,
    helmholtz_residual_1d,
    make_heat_1d_loss,
    make_helmholtz_1d_loss,
    make_poisson_1d_loss,
    poisson_1d_source,
    poisson_residual_1d,
)
from jax_pirfn.train import adam_init, jacobian_map, ridge_readout, train_readout, train_step


def test_cosine_identity():
    x = jnp.array([[0.2], [0.5], [0.8]])
    omega = jnp.array([[1.5], [-0.7], [3.0]])
    b = jnp.array([0.1, 1.2, 2.3])
    phi = cosine_features(x, omega, b)
    expected = jnp.cos(x @ omega.T + b)
    np.testing.assert_allclose(np.asarray(phi), np.asarray(expected), rtol=1e-6)


def test_sampling_shapes():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    n, d = 11, 3
    g = sample_omega(keys[0], n, d, kernel="gaussian", sigma=0.7)
    m = sample_omega(keys[1], n, d, kernel="matern", sigma=1.2, nu=2.5)
    lap = sample_omega(keys[2], n, d, kernel="laplace", sigma=0.8)
    b = sample_bias(keys[3], n)
    assert g.shape == (n, d)
    assert m.shape == (n, d)
    assert lap.shape == (n, d)
    assert b.shape == (n,)
    assert jnp.all(b >= 0.0) and jnp.all(b <= 2.0 * jnp.pi)


def test_product_features_factor():
    key = jax.random.PRNGKey(1)
    params = init_product_features(key, n_x=4, n_t=5, sigma_x=0.6, sigma_t=1.1)
    x = jnp.linspace(0.0, 1.0, 7)
    t = jnp.linspace(0.0, 0.5, 7)
    phi = product_features(
        x, t, params["omega_x"], params["b_x"], params["omega_t"], params["b_t"]
    )
    px = cosine_features(x, params["omega_x"], params["b_x"])
    pt = cosine_features(t, params["omega_t"], params["b_t"])
    expected = (px[:, :, None] * pt[:, None, :]).reshape(7, 20)
    np.testing.assert_allclose(np.asarray(phi), np.asarray(expected), rtol=1e-6)
    u = apply_product(
        x, t, params["omega_x"], params["b_x"], params["omega_t"], params["b_t"], params["c"]
    )
    assert u.shape == (7,)


def test_frozen_features_only_c_changes():
    key = jax.random.PRNGKey(2)
    params = init_random_features(key, n_features=16, dim=1, sigma=0.5)
    x_int = jnp.linspace(0.05, 0.95, 20)
    _, f = poisson_1d_source(x_int)
    x_bd = jnp.array([0.0, 1.0])
    g_bd = jnp.zeros((2,))
    omega = params["omega"]
    b = params["b"]
    c = params["c"]
    omega_0 = jnp.array(omega)
    b_0 = jnp.array(b)
    c_0 = jnp.array(c)
    loss_fn = make_poisson_1d_loss(omega, b, x_int, f, x_bd, g_bd)
    c_new, frozen, _, _ = train_step(c, {"omega": omega, "b": b}, loss_fn, adam_init(c), lr=1e-2)
    np.testing.assert_allclose(np.asarray(frozen["omega"]), np.asarray(omega_0))
    np.testing.assert_allclose(np.asarray(frozen["b"]), np.asarray(b_0))
    np.testing.assert_allclose(np.asarray(omega), np.asarray(omega_0))
    np.testing.assert_allclose(np.asarray(b), np.asarray(b_0))
    assert not jnp.allclose(c_new, c_0)


def test_poisson_manufactured_relative_l2():
    key = jax.random.PRNGKey(3)
    n_feat = 96
    params = init_random_features(key, n_features=n_feat, dim=1, kernel="gaussian", sigma=0.45)
    x_int = jnp.linspace(0.0, 1.0, 80)[1:-1]
    # pad to ~80 collocation points
    x_extra = jnp.array([0.01, 0.99])
    x_int = jnp.concatenate([x_int, x_extra])
    _, f = poisson_1d_source(x_int)
    x_bd = jnp.array([0.0, 1.0])
    g_bd = jnp.zeros((2,))
    loss_fn = make_poisson_1d_loss(
        params["omega"], params["b"], x_int, f, x_bd, g_bd, lam_l2=1e-6
    )
    c, _ = train_readout(params["c"], loss_fn, n_steps=400, lr=2e-2)
    x_test = jnp.linspace(0.0, 1.0, 200)
    u_true, _ = poisson_1d_source(x_test)
    u_hat = apply_1d(x_test, params["omega"], params["b"], c)
    err = float(relative_l2(u_hat, u_true))
    assert err < 0.15, f"relative L2 {err} >= 0.15"


def test_helmholtz_residual_beats_random_c():
    key = jax.random.PRNGKey(4)
    kappa = 8.0
    params = init_random_features(key, n_features=64, dim=1, kernel="gaussian", sigma=0.45)
    x_int = jnp.linspace(0.05, 0.95, 60)
    _, f = helmholtz_1d_source(x_int, kappa=kappa)
    x_bd = jnp.array([0.0, 1.0])
    g_bd = jnp.zeros((2,))
    loss_fn = make_helmholtz_1d_loss(
        params["omega"], params["b"], x_int, f, x_bd, g_bd, kappa=kappa, lam_l2=1e-6
    )
    c_tr, _ = train_readout(params["c"], loss_fn, n_steps=250, lr=2e-2)
    c_rand = jax.random.normal(jax.random.PRNGKey(99), params["c"].shape)
    r_tr = helmholtz_residual_1d(x_int, params["omega"], params["b"], c_tr, f, kappa=kappa)
    r_rand = helmholtz_residual_1d(x_int, params["omega"], params["b"], c_rand, f, kappa=kappa)
    assert float(jnp.mean(r_tr ** 2)) < float(jnp.mean(r_rand ** 2))


def test_heat_product_error_or_terms_drop():
    key = jax.random.PRNGKey(5)
    params = init_product_features(
        key, n_x=16, n_t=16, sigma_x=0.5, sigma_t=1.5, kernel_x="gaussian", kernel_t="gaussian"
    )
    ox, bx, ot, bt = params["omega_x"], params["b_x"], params["omega_t"], params["b_t"]
    xs = jnp.linspace(0.0, 1.0, 24)
    ts = jnp.linspace(0.0, 0.2, 16)
    xx, tt = jnp.meshgrid(xs[1:-1], ts[1:], indexing="ij")
    x_int, t_int = xx.reshape(-1), tt.reshape(-1)
    x_ic = jnp.linspace(0.0, 1.0, 40)
    t_ic = jnp.zeros_like(x_ic)
    u_ic = heat_1d_exact(x_ic, t_ic)
    t_bc = jnp.linspace(0.0, 0.2, 24)
    x_bc = jnp.concatenate([jnp.zeros_like(t_bc), jnp.ones_like(t_bc)])
    t_bc_all = jnp.concatenate([t_bc, t_bc])
    u_bc = jnp.zeros_like(x_bc)

    def parts(c):
        r = heat_residual(x_int, t_int, ox, bx, ot, bt, c)
        ic = apply_product(x_ic, t_ic, ox, bx, ot, bt, c) - u_ic
        bc = apply_product(x_bc, t_bc_all, ox, bx, ot, bt, c) - u_bc
        return loss_terms(r, bc, c, ic_residual=ic)

    init_terms = parts(params["c"])
    loss_fn = make_heat_1d_loss(
        ox, bx, ot, bt, x_int, t_int, x_ic, t_ic, u_ic, x_bc, t_bc_all, u_bc, lam_l2=1e-6
    )
    c_adam, _ = train_readout(params["c"], loss_fn, n_steps=80, lr=2e-2)
    n = params["c"].shape[0]
    a_pde = jacobian_map(lambda c: heat_residual(x_int, t_int, ox, bx, ot, bt, c), n)
    a_ic = jacobian_map(lambda c: apply_product(x_ic, t_ic, ox, bx, ot, bt, c), n)
    a_bc = jacobian_map(lambda c: apply_product(x_bc, t_bc_all, ox, bx, ot, bt, c), n)
    c = ridge_readout([(a_pde, jnp.zeros(a_pde.shape[0])), (a_ic, u_ic), (a_bc, u_bc)], lam_l2=1e-6)
    trained_terms = parts(c)
    xx, tt = jnp.meshgrid(jnp.linspace(0.0, 1.0, 25), jnp.linspace(0.0, 0.2, 25), indexing="ij")
    x_te = xx.reshape(-1)
    t_te = tt.reshape(-1)
    u_true = heat_1d_exact(x_te, t_te)
    u_hat = apply_product(x_te, t_te, ox, bx, ot, bt, c)
    err = float(relative_l2(u_hat, u_true))
    dropped = (
        float(trained_terms["pde"]) <= float(init_terms["pde"])
        and float(trained_terms["ic"]) < float(init_terms["ic"])
        and float(trained_terms["bc"]) <= float(init_terms["bc"]) + 1e-8
    )
    assert err < 0.25 or dropped, f"heat L2={err}, terms init={init_terms} trained={trained_terms}"
    _ = c_adam


def test_residual_grad_finite_and_loss_nonneg():
    key = jax.random.PRNGKey(6)
    params = init_random_features(key, n_features=32, dim=1, sigma=0.5, init_c="random")
    x_int = jnp.linspace(0.1, 0.9, 16)
    _, f = poisson_1d_source(x_int)
    x_bd = jnp.array([0.0, 1.0])
    g_bd = jnp.zeros((2,))
    r = poisson_residual_1d(x_int, params["omega"], params["b"], params["c"], f)
    bc = apply_1d(x_bd, params["omega"], params["b"], params["c"]) - g_bd
    terms = loss_terms(r, bc, params["c"])
    for name, val in terms.items():
        assert float(val) >= -1e-12, f"{name} negative: {val}"
    loss_fn = make_poisson_1d_loss(
        params["omega"], params["b"], x_int, f, x_bd, g_bd
    )
    g = jax.grad(loss_fn)(params["c"])
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(pirfn_loss(r, bc, params["c"])) >= 0.0
