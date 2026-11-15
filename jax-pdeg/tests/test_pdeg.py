import jax.numpy as jnp
import numpy as np

from jax_pdeg import (
    certificate_S,
    init_state,
    params_from_L_mu_tau,
    project_box,
    quadratic_scsc,
    run_pdeg,
    run_pdeg_warmup,
    step,
)


def _scsc_scalar():
    """f(x,y) = (1/2) x^2 + x y - (1/2) y^2 on R x R."""
    return quadratic_scsc(mu=1.0)


def test_one_step_matches_hand_computed_unconstrained():
    prob = _scsc_scalar()
    p = params_from_L_mu_tau(L=1.0, mu=1.0, tau=0.0)
    x0 = jnp.array([1.0])
    y0 = jnp.array([0.5])
    st = init_state(x0, y0)
    st1, S, _, nq = step(
        st, prob.grad_x, prob.grad_y, p, lambda u: u, lambda u: u, cached_fgrad=None
    )
    assert nq == 3

    # Independent numpy replay of Algorithm 1, t=0, ξ=n=v=0, z=x.
    lam, tau, h, alpha, gamma = p.lam, p.tau, p.h, p.alpha, p.gamma
    omag = (1.0 - alpha) * gamma
    x = np.array([1.0])
    y = np.array([0.5])
    z = x.copy()
    xi = np.zeros(1)
    n = np.zeros(1)
    v = np.zeros(1)
    gx = x + y
    gy = x - y
    gxG = gx + lam * (x - z)
    gyG = gy - tau * y
    x_tilde = x - h * (gxG + xi)
    y_tilde = y + h * (gyG - n + v)
    gx_t = x_tilde + y_tilde
    gy_t = x_tilde - y_tilde
    gxG_t = gx_t + lam * (x_tilde - z)
    gyG_t = gy_t - tau * y_tilde
    vbar = alpha * v + omag * gyG_t
    x_new = x - h * gxG_t
    xi_new = (x - x_new) / h - gxG_t
    y_new = y + h * (gyG_t + vbar)
    n_new = ((y - y_new) / h + gyG_t + vbar) / (1.0 + omag)
    gy_c = x_new - y_new
    gyG_c = gy_c - tau * y_new
    v_new = alpha * v + omag * (gyG_c - n_new)
    z_new = z + p.beta * (x_new - z)
    gx_c = x_new + y_new
    S_hand = np.dot(gx_c + xi_new, gx_c + xi_new) + np.dot(
        -gy_c + tau * y_new + n_new, -gy_c + tau * y_new + n_new
    )

    np.testing.assert_allclose(np.asarray(st1.x), x_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(st1.y), y_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(st1.z), z_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(st1.xi), xi_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(st1.n), n_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(st1.v), v_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(S), S_hand, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(st1.xi), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st1.n), 0.0, atol=1e-12)


def test_lemma_31_normal_cone_on_box():
    # Bilinear on a box so the dual hits the boundary.
    A = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float64)

    def f(x, y):
        return jnp.dot(x, A @ y)

    def gx(x, y):
        return A @ y

    def gy(x, y):
        return A.T @ x

    lo, hi = -0.2, 0.2
    proj = lambda u: project_box(u, lo, hi)
    p = params_from_L_mu_tau(L=2.0, mu=0.0, tau=0.5)
    x0 = jnp.array([0.15, -0.15])
    y0 = jnp.array([0.0, 0.1])
    res = run_pdeg(gx, gy, x0, y0, p, project_x=proj, project_y=proj, T=40, os_index=0)
    rng = np.random.default_rng(1)
    for t in range(res.x_hist.shape[0]):
        x = np.asarray(res.x_hist[t])
        y = np.asarray(res.y_hist[t])
        xi = np.asarray(res.xi_hist[t])
        n = np.asarray(res.n_hist[t])
        assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
        assert np.all(y >= lo - 1e-12) and np.all(y <= hi + 1e-12)
        for _ in range(8):
            xp = rng.uniform(lo, hi, size=x.shape)
            yp = rng.uniform(lo, hi, size=y.shape)
            assert np.dot(xi, xp - x) <= 1e-10
            assert np.dot(n, yp - y) <= 1e-10
        tol = 1e-8
        for i in range(x.size):
            if x[i] >= hi - tol:
                assert xi[i] >= -1e-8
            elif x[i] <= lo + tol:
                assert xi[i] <= 1e-8
            else:
                assert abs(xi[i]) <= 1e-8
        for i in range(y.size):
            if y[i] >= hi - tol:
                assert n[i] >= -1e-8
            elif y[i] <= lo + tol:
                assert n[i] <= 1e-8
            else:
                assert abs(n[i]) <= 1e-8


def test_z_update_and_warmup_freeze():
    prob = quadratic_scsc(mu=1.0)
    p = params_from_L_mu_tau(L=np.sqrt(2.0), mu=1.0, tau=0.0)
    x0 = jnp.array([1.0, -0.5])
    y0 = jnp.array([0.3, 0.2])
    T_w, T = 5, 4
    res = run_pdeg_warmup(
        prob.grad_x,
        prob.grad_y,
        x0,
        y0,
        p,
        T=T,
        T_w=T_w,
        os_index=T_w,
    )
    z_hist = np.asarray(res.z_hist)
    x_hist = np.asarray(res.x_hist)
    np.testing.assert_allclose(z_hist[0], np.asarray(x0))
    for t in range(T_w):
        np.testing.assert_allclose(z_hist[t + 1], z_hist[0], atol=1e-14)
    for t in range(T_w, T_w + T):
        expected = (1.0 - p.beta) * z_hist[t] + p.beta * x_hist[t + 1]
        np.testing.assert_allclose(z_hist[t + 1], expected, rtol=1e-12, atol=1e-14)


def test_certificate_nonnegative_and_scsc_does_not_explode():
    mu = 1.0
    L = float(np.sqrt(2.0))
    prob = quadratic_scsc(mu=mu)
    p = params_from_L_mu_tau(L=L, mu=mu, tau=0.0)
    x0 = jnp.array([1.0, -0.7])
    y0 = jnp.array([0.4, 0.2])
    res = run_pdeg(prob.grad_x, prob.grad_y, x0, y0, p, T=250, os_index=0)
    S = np.asarray(res.S)
    assert np.all(S >= -1e-14)
    assert np.all(np.isfinite(S))
    xs = np.asarray(res.x_hist)
    ys = np.asarray(res.y_hist)
    zs = np.asarray(res.z_hist)
    assert np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))

    # Inner saddle of G(·,·; z): x*(z) = λ z / (1 + 1/μ + λ), y* = x*/μ.
    lam = p.lam
    xstar = (lam / (1.0 + 1.0 / mu + lam)) * zs
    ystar = xstar / mu
    err = np.linalg.norm(xs - xstar, axis=1) + np.linalg.norm(ys - ystar, axis=1)
    assert err[-1] < err[0]
    assert err[-1] < 0.5 * err[0] + 0.5
    # Outer distance should not explode.
    dist0 = np.linalg.norm(xs[0]) + np.linalg.norm(ys[0])
    distT = np.linalg.norm(xs[-1]) + np.linalg.norm(ys[-1])
    assert distT < 5.0 * dist0 + 1.0
    assert S[-1] < 10.0 * S[0] + 1.0


def test_output_rules_alg1_and_alg2():
    prob = quadratic_scsc(mu=1.0)
    p = params_from_L_mu_tau(L=np.sqrt(2.0), mu=1.0, tau=0.0)
    x0 = jnp.array([0.8, -0.2])
    y0 = jnp.array([0.1, 0.4])
    T = 12
    res = run_pdeg(prob.grad_x, prob.grad_y, x0, y0, p, T=T, os_index=3)
    assert res.gs_index == int(np.argmin(np.asarray(res.S)))
    assert 0 <= res.os_index < T
    np.testing.assert_allclose(res.x_out, res.x_hist[res.gs_index + 1])
    np.testing.assert_allclose(res.y_out, res.y_hist[res.gs_index + 1])
    np.testing.assert_allclose(res.z_out, res.z_hist[res.os_index])
    S_gs = certificate_S(
        prob.grad_x,
        prob.grad_y,
        res.x_out,
        res.y_out,
        res.xi_hist[res.gs_index + 1],
        res.n_hist[res.gs_index + 1],
        p.tau,
    )
    np.testing.assert_allclose(S_gs, res.S[res.gs_index], rtol=1e-12)

    T_w = 6
    res2 = run_pdeg_warmup(
        prob.grad_x, prob.grad_y, x0, y0, p, T=T, T_w=T_w, os_index=T_w + 2
    )
    assert T_w <= res2.os_index < T_w + T
    window = np.asarray(res2.S)[T_w : T_w + T]
    assert res2.gs_index == T_w + int(np.argmin(window))
    assert res2.T_w == T_w


def test_oracle_call_count_cached():
    calls = {"n": 0}

    def gx(x, y):
        calls["n"] += 1
        return x + y

    def gy(x, y):
        return x - y

    p = params_from_L_mu_tau(L=1.0, mu=1.0, tau=0.0)
    T = 7
    res = run_pdeg(gx, gy, jnp.array([1.0]), jnp.array([0.2]), p, T=T, os_index=0)
    # One full first-order query counted on grad_x (grad_y is free in this wrap).
    assert calls["n"] == 1 + 2 * T
    assert res.n_oracle == 1 + 2 * T


def test_ncc_interval_runs():
    from jax_pdeg import ncc_interval

    prob = ncc_interval(
        project_x=lambda u: project_box(u, -2.0, 2.0),
        project_y=lambda u: project_box(u, -1.0, 1.0),
    )
    # L of ∇f: f = -x^2/4 + x y, ∇x = -x/2 + y, ∇y = x; Lipschitz ~ 1.
    p = params_from_L_mu_tau(L=1.0, mu=0.0, tau=0.5)
    res = run_pdeg(
        prob.grad_x,
        prob.grad_y,
        jnp.array([0.5]),
        jnp.array([0.2]),
        p,
        project_x=prob.project_x,
        project_y=prob.project_y,
        T=20,
        os_index=0,
    )
    assert np.all(np.asarray(res.S) >= -1e-14)
    assert np.all(np.abs(np.asarray(res.y_hist)) <= 1.0 + 1e-12)
