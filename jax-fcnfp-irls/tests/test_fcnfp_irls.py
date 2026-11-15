"""CPU tests for arXiv:2609.09295 IRLS (graphs with n<=8, m<=20)."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from jax_fcnfp_irls import (
    conservation_residual,
    dual_ssn,
    free_arc_mask,
    irls,
    irls_weights,
    j_true,
    ll_derivative,
    ll_log_tail,
    ll_middle,
    ll_quadratic,
    ll_second_derivative,
    ll_value,
    log_tail_alpha,
    mu_max,
    newton_diag,
    newton_matrix,
    node_arc_incidence,
    primal_from_dual,
    transition_params,
)


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _admissible_mu(r, b, lam, alpha, frac=0.5):
    cap = np.min(_f64(mu_max(r, b, lam, alpha)))
    mu = min(frac * lam, cap)
    return max(min(mu, 0.99 * lam), 1e-16)


# ---------------------------------------------------------------------------
# C1: LL value and derivative match at kinks
# ---------------------------------------------------------------------------


def test_c1_ll_kink_matching():
    rng = np.random.default_rng(0)
    for _ in range(24):
        r = float(rng.uniform(0.05, 3.0))
        b = float(rng.uniform(0.2, 6.0))
        lam = float(rng.uniform(0.2, 3.0))
        lam0 = float(rng.uniform(0.2, 3.0))
        alpha = float(log_tail_alpha(lam, lam0))
        mu = _admissible_mu(r, b, lam, alpha, frac=0.4)

        T, x1, S, x2, psi2 = transition_params(r, b, lam, mu, alpha)
        T, x1, x2, psi2 = map(float, (T, x1, x2, psi2))
        assert x1 > 0.0 and x2 >= x1 - 1e-12

        # Values: quadratic and middle match at x1
        q1 = float(ll_quadratic(x1, lam, mu))
        m1 = float(ll_middle(x1, T, lam, mu))
        v1 = float(ll_value(x1, r, b, lam, mu, alpha))
        assert q1 == pytest.approx(m1, rel=1e-9, abs=1e-10)
        assert v1 == pytest.approx(q1, rel=1e-9, abs=1e-10)

        # Derivatives at x1 both equal T/lam
        d_quad = x1 / (lam - mu)
        d_mid = (T - x1) / mu
        target = T / lam
        assert d_quad == pytest.approx(target, rel=1e-9, abs=1e-10)
        assert d_mid == pytest.approx(target, rel=1e-9, abs=1e-10)
        d1 = float(ll_derivative(x1, r, b, lam, mu, alpha))
        assert d1 == pytest.approx(target, rel=1e-9, abs=1e-10)

        # Values: middle and log-tail match at x2
        m2 = float(ll_middle(x2, T, lam, mu))
        t2 = float(ll_log_tail(x2, r, b, x2, psi2, alpha))
        v2 = float(ll_value(x2, r, b, lam, mu, alpha))
        assert m2 == pytest.approx(t2, rel=1e-8, abs=1e-9)
        assert v2 == pytest.approx(m2, rel=1e-8, abs=1e-9)

        d_mid_x2 = (T - x2) / mu
        d_tail_x2 = r + alpha * b / x2
        assert d_mid_x2 == pytest.approx(d_tail_x2, rel=1e-7, abs=1e-8)
        d2 = float(ll_derivative(x2, r, b, lam, mu, alpha))
        assert d2 == pytest.approx(d_tail_x2, rel=1e-7, abs=1e-8)


# ---------------------------------------------------------------------------
# C2: IRLS weights positive, finite, clipped; f=0 uses 1/(lam-mu)
# ---------------------------------------------------------------------------


def test_c2_irls_weights():
    r = np.array([0.5, 1.2, 0.0, 3.0])
    b = np.array([1.0, 2.0, 4.0, 0.5])
    lam = 0.8
    lam0 = 1.0
    alpha = float(log_tail_alpha(lam, lam0))
    mu = _admissible_mu(r, b, lam, alpha)

    T, x1, _S, x2, _psi2 = transition_params(r, b, lam, mu, alpha)
    T = _f64(T)
    x1 = _f64(x1)
    x2 = _f64(x2)

    f = np.array([0.0, 0.5 * x1[1], 0.5 * (x1[2] + x2[2]), 2.0 * x2[3] + 1.0])
    w = _f64(irls_weights(f, r, b, lam, mu, alpha, clip=True))
    assert np.all(np.isfinite(w))
    assert np.all(w >= 1e-12 - 1e-18)
    assert np.all(w <= 1e12 + 1e-6)

    # f=0 uses 1/(lam-mu)
    assert w[0] == pytest.approx(1.0 / (lam - mu), rel=1e-12)

    # log-tail formula
    ax = abs(f[3])
    w_tail = r[3] / ax + alpha * b[3] / ax**2
    assert w[3] == pytest.approx(w_tail, rel=1e-10)

    # clipping: a huge tail weight is clipped
    f_tiny = np.array([1e-30, 1e-30, 1e-30, 1e-30])
    # put one coordinate deep in the tail of a large-b edge by using x2 small...
    # instead force clip by evaluating unclipped vs clipped at a huge |f| wait:
    # huge |f| makes tail weight small. Tiny |f| in tail: need x2 tiny.
    w_unclip = _f64(irls_weights(np.zeros(4), r, b, lam, mu, alpha, clip=False))
    w_clip = _f64(irls_weights(np.zeros(4), r, b, lam, mu, alpha, clip=True))
    assert np.all(w_clip >= 1e-12) and np.all(w_clip <= 1e12)
    assert np.all(w_unclip > 0) and np.all(np.isfinite(w_unclip))


# ---------------------------------------------------------------------------
# C3: weight vs curvature
# ---------------------------------------------------------------------------


def test_c3_weight_vs_curvature():
    r, b, lam, lam0 = 0.7, 1.5, 1.1, 1.1
    alpha = float(log_tail_alpha(lam, lam0))
    mu = _admissible_mu(r, b, lam, alpha, frac=0.45)
    T, x1, _S, x2, _psi2 = map(float, transition_params(r, b, lam, mu, alpha))

    # quadratic region: w = 1/(lam-mu) = LL''(x)
    xq = 0.5 * x1
    wq = float(irls_weights(xq, r, b, lam, mu, alpha, clip=False))
    assert wq == pytest.approx(1.0 / (lam - mu), rel=1e-12)
    d2 = float(ll_second_derivative(xq, r, b, lam, mu, alpha))
    assert d2 == pytest.approx(1.0 / (lam - mu), rel=1e-12)
    assert wq == pytest.approx(d2, rel=1e-12)

    # middle region: paper weight formula (not equal to LL'' = -1/mu)
    if x2 > x1 + 1e-9:
        xm = 0.5 * (x1 + x2)
        wm = float(irls_weights(xm, r, b, lam, mu, alpha, clip=False))
        assert wm == pytest.approx((T - xm) / (mu * xm), rel=1e-10)
        d2m = float(ll_second_derivative(xm, r, b, lam, mu, alpha))
        assert d2m == pytest.approx(-1.0 / mu, rel=1e-12)


# ---------------------------------------------------------------------------
# C4: incidence conservation
# ---------------------------------------------------------------------------


def test_c4_incidence_conservation():
    # 4-node diamond: 0 -> {1,2} -> 3
    tails = [0, 1, 0, 2]
    heads = [1, 3, 2, 3]
    C = node_arc_incidence(tails, heads, n_nodes=4)
    f = jnp.asarray([0.4, 0.4, 0.6, 0.6], dtype=jnp.float64)
    d = C @ f
    res = conservation_residual(C, f, d)
    assert float(jnp.linalg.norm(res)) == pytest.approx(0.0, abs=1e-14)

    # C_{i,e}=+1 if e leaves i, -1 if e enters i
    C_np = _f64(C)
    assert C_np[0, 0] == 1.0 and C_np[1, 0] == -1.0
    assert C_np[1, 1] == 1.0 and C_np[3, 1] == -1.0


# ---------------------------------------------------------------------------
# C5: dual map bounds; Newton matrix symmetric PSD with 1-kernel
# ---------------------------------------------------------------------------


def test_c5_dual_map_and_newton_matrix():
    tails = [0, 1]
    heads = [1, 2]
    C = node_arc_incidence(tails, heads, n_nodes=3)
    c = jnp.asarray([2.0, 2.0])
    w = jnp.asarray([1.5, 0.8])
    d = jnp.asarray([1.0, 0.0, -1.0])
    f, y, res, _k = dual_ssn(C, d, c, w)
    fy = primal_from_dual(y, C, w, c)
    fy_np = _f64(fy)
    assert np.all(fy_np >= -1e-12)
    assert np.all(fy_np <= _f64(c) + 1e-12)

    D = newton_diag(y, C, w, c)
    L = _f64(newton_matrix(C, D))
    assert np.allclose(L, L.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(0.5 * (L + L.T))
    assert np.min(eigs) >= -1e-8

    ones = np.ones(L.shape[0])
    # kernel contains all-ones when the free subgraph is connected
    mask = _f64(free_arc_mask(y, C, w, c))
    if np.all(mask > 0.5):
        assert np.linalg.norm(L @ ones) <= 1e-8
    else:
        # still 1^T C = 0 so L 1 = 0 regardless of D
        assert np.linalg.norm(L @ ones) <= 1e-8


# ---------------------------------------------------------------------------
# C6: Dual SSN on a tiny feasible instance
# ---------------------------------------------------------------------------


def test_c6_dual_ssn_tiny_instance():
    # 4-node diamond: cheap high-fixed path 0-1-3 vs expensive low-fixed 0-2-3
    tails = [0, 1, 0, 2]
    heads = [1, 3, 2, 3]
    C = node_arc_incidence(tails, heads, n_nodes=4)
    d = jnp.asarray([1.0, 0.0, 0.0, -1.0])
    c = jnp.asarray([2.0, 2.0, 2.0, 2.0])
    w = jnp.asarray([0.2, 0.2, 5.0, 5.0])
    f, y, res, n_iters = dual_ssn(C, d, c, w)
    f_np = _f64(f)
    assert res < 1e-5
    assert float(jnp.linalg.norm(C @ f - d, ord=jnp.inf)) < 1e-5
    assert np.all(f_np >= -1e-8)
    assert np.all(f_np <= _f64(c) + 1e-8)
    assert n_iters <= 40

    # 3-node path (unique feasible flow)
    Cpath = node_arc_incidence([0, 1], [1, 2], n_nodes=3)
    dpath = jnp.asarray([1.0, 0.0, -1.0])
    cpath = jnp.asarray([2.0, 2.0])
    wpath = jnp.asarray([1.0, 3.0])
    fp, _yp, resp, _ = dual_ssn(Cpath, dpath, cpath, wpath)
    assert resp < 1e-5
    assert np.allclose(_f64(fp), [1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# C7: end-to-end IRLS on two parallel arcs
# ---------------------------------------------------------------------------


def test_c7_irls_end_to_end():
    # Two parallel arcs 0 -> 1, demand 1. Arc 0: huge b, tiny r (bad).
    # Arc 1: tiny b, large r (good). Demand fits either.
    tails = [0, 0]
    heads = [1, 1]
    C = node_arc_incidence(tails, heads, n_nodes=2)
    d = jnp.asarray([1.0, -1.0])
    c = jnp.asarray([1.5, 1.5])
    r = jnp.asarray([0.001, 5.0])
    b = jnp.asarray([50.0, 0.01])
    f0 = jnp.asarray([0.5, 0.5])

    result = irls(C, d, c, r, b, f0=f0, start="dense", max_iters=80, do_refit=True)
    f = _f64(result.flow)
    assert result.feasible
    assert float(jnp.linalg.norm(C @ result.flow - d, ord=jnp.inf)) < 1e-4
    assert np.all(f >= -1e-6) and np.all(f <= _f64(c) + 1e-6)
    assert np.isfinite(result.j_true)

    f_bad = jnp.asarray([1.0, 0.0])
    j_bad = float(j_true(f_bad, r, b, tau=1e-6))
    assert result.j_true <= j_bad + 1e-8

    hist = result.incumbent_history
    assert len(hist) >= 1
    for a, bcost in zip(hist, hist[1:]):
        assert bcost <= a + 1e-8


# ---------------------------------------------------------------------------
# C8: mu_max formula
# ---------------------------------------------------------------------------


def test_c8_mu_max_formula():
    lam = 1.3
    alpha = float(log_tail_alpha(lam, 0.9))
    # r = 0 reduces to lam / (2 alpha)
    b = 2.4
    mm = float(mu_max(0.0, b, lam, alpha))
    assert mm == pytest.approx(lam / (2.0 * alpha), rel=1e-12, abs=1e-12)

    rng = np.random.default_rng(1)
    for _ in range(20):
        r = float(rng.uniform(0.05, 2.0))
        b = float(rng.uniform(0.2, 5.0))
        lam = float(rng.uniform(0.3, 2.5))
        alpha = float(log_tail_alpha(lam, lam))
        mm = float(mu_max(r, b, lam, alpha))
        T = lam * r + np.sqrt(2.0 * lam * b)
        S = T - mm * r
        disc = S * S - 4.0 * mm * alpha * b
        assert disc >= -1e-8
