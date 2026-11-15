"""Tests for the ring-plus-chord HyperQ readout."""

import math

import jax
import jax.numpy as jnp
import pytest

from hyperq.readout import edges, emit, init_params, readout, residual


def _degrees(edge_list, n):
    counts = [0] * n
    for u, v in edge_list.tolist():
        counts[u] += 1
        counts[v] += 1
    return counts


def test_edge_degree_and_count():
    for n, c in ((8, 3), (16, 3)):
        assert math.gcd(c, n) == 1
        graph = edges(n, c)
        edge_list, incident = graph.edge_list, graph.incident
        assert edge_list.shape == (2 * n, 2)
        assert incident.shape == (n, 4)
        assert len({frozenset(e) for e in edge_list.tolist()}) == 2 * n
        assert _degrees(edge_list, n) == [4] * n
        for q in range(n):
            idxs = incident[q].tolist()
            assert len(set(idxs)) == 4
            for eidx in idxs:
                assert q in edge_list[eidx].tolist()


def test_emit_ranges_and_unit_psi():
    n, d_h, d_a = 8, 5, 4
    params = init_params(jax.random.PRNGKey(0), n, d_h, d_a)
    assert params["W_omega"].shape == (2 * n, d_a)
    assert params["W_psi"].shape == (2 * n, d_a)
    assert "b" not in params and "yz" not in params
    h = jax.random.normal(jax.random.PRNGKey(1), (d_h,))
    alpha, omega, psi = emit(h, params)
    psi_y, psi_z = psi
    assert alpha.shape == (n,)
    assert omega.shape == (2 * n,)
    assert jnp.all(alpha > -jnp.pi) and jnp.all(alpha < jnp.pi)
    assert jnp.all(omega > -1.2) and jnp.all(omega < 1.2)
    assert psi_y.shape == (n,) and psi_z.shape == (n,)
    assert jnp.allclose(psi_y**2 + psi_z**2, 1.0)

    b = params["W_A"] @ h
    assert jnp.allclose(alpha, jnp.pi * jnp.tanh(params["W_alpha"] @ b))
    assert jnp.allclose(omega, 1.2 * jnp.tanh(params["W_omega"] @ b))
    raw = (params["W_psi"] @ b).reshape((n, 2))
    nrm = jnp.sqrt(jnp.sum(raw**2, axis=1, keepdims=True))
    expect = raw / nrm
    assert jnp.allclose(psi_y, expect[:, 0])
    assert jnp.allclose(psi_z, expect[:, 1])


def test_identity_and_product_class():
    n, c = 8, 3
    incident = edges(n, c).incident
    psi_y = jnp.array([0.6, -0.8, 0.0, 1.0, -1.0, 0.3, -0.4, 0.0])
    psi_z = jnp.array([0.8, 0.6, 1.0, 0.0, 0.0, 0.4, 0.5, -1.0])
    alpha = jnp.array([0.0, 0.3, -0.7, 1.1, -1.4, 0.2, 0.5, -0.4])
    omega = jnp.zeros((2 * n,))
    z = readout(alpha, omega, (psi_y, psi_z), incident)
    # omega = 0 drops the edge product, including the identity alpha = 0.
    expect = psi_z * jnp.cos(alpha) - psi_y * jnp.sin(alpha)
    assert jnp.allclose(z, expect)
    assert jnp.allclose(z[0], psi_z[0])


def test_hand_computed_numeric_product():
    n, c = 8, 3
    graph = edges(n, c)
    incident = graph.incident
    edge_list = graph.edge_list
    # Qubit 0, four incident edges. alpha = 0, psi = (0, 1) so a = 1.
    # Couplings pi/3, pi/3, 0, 0 -> cosines 1/2, 1/2, 1, 1.
    # Product = 0.25, so z_0 = 0.25. Lookup is by edge index, not neighbor id.
    q = 0
    e0, e1, e2, e3 = [int(x) for x in incident[q].tolist()]
    omega = jnp.zeros((2 * n,))
    omega = omega.at[e0].set(jnp.pi / 3)
    omega = omega.at[e1].set(jnp.pi / 3)
    alpha = jnp.zeros((n,))
    psi = (jnp.zeros((n,)), jnp.ones((n,)))
    z = readout(alpha, omega, psi, incident)
    assert z[0] == pytest.approx(0.25)
    # Nontrivial amplitude: a = 0.8 * cos(0) - 0.6 * sin(0) = 0.8, times 0.25.
    psi_y = jnp.zeros((n,)).at[0].set(0.6)
    psi_z = jnp.ones((n,)).at[0].set(0.8)
    z2 = readout(alpha, omega, (psi_y, psi_z), incident)
    assert z2[0] == pytest.approx(0.2)
    nbr = []
    for eidx in (e0, e1, e2, e3):
        u, v = [int(x) for x in edge_list[eidx].tolist()]
        nbr.append(v if u == q else u)
    wrong = 1.0
    for j in nbr:
        wrong = wrong * jnp.cos(omega[j])
    assert wrong != pytest.approx(0.25)


def _partials(a, omega, e, cos_fn):
    cos = [cos_fn(omega[e[k]]) for k in range(4)]
    sin = [jnp.sin(omega[e[k]]) for k in range(4)]
    out = []
    for k in range(4):
        others = jnp.array(1.0)
        for j in range(4):
            if j != k:
                others = others * cos[j]
        out.append(-a * sin[k] * others)
    return out


def test_grad_finite_matches_analytic_when_cosine_is_zero():
    import importlib

    ro = importlib.import_module("hyperq.readout")

    n, c = 8, 3
    incident = edges(n, c).incident
    q = 0
    e = [int(x) for x in incident[q].tolist()]
    alpha = jnp.zeros((n,)).at[q].set(0.4)
    psi_y = jnp.zeros((n,)).at[q].set(0.6)
    psi_z = jnp.ones((n,)).at[q].set(0.8)
    a = psi_z[q] * jnp.cos(alpha[q]) - psi_y[q] * jnp.sin(alpha[q])

    def z_q(w):
        return readout(alpha, w, (psi_y, psi_z), incident)[q]

    # Near the cosine zero (float cos(pi/2) is a tiny residual, not exact).
    omega = jnp.zeros((2 * n,))
    omega = omega.at[e[0]].set(jnp.pi / 2)
    omega = omega.at[e[1]].set(jnp.pi / 3)
    omega = omega.at[e[2]].set(jnp.pi / 6)
    omega = omega.at[e[3]].set(0.0)
    g = jax.grad(z_q)(omega)
    assert jnp.all(jnp.isfinite(g))
    for k, analytic in enumerate(_partials(a, omega, e, jnp.cos)):
        assert g[e[k]] == pytest.approx(analytic, rel=1e-5, abs=1e-6)

    # Exact cosine 0. libm cos never returns 0, so the test cosine snaps one
    # coupling to 0 while keeping the derivative -sin. Four multiplies must
    # still yield -a*sin(omega_e)*prod(other cosines), not a 0/0.
    sentinel = jnp.array(0.123456, dtype=omega.dtype)
    orig = ro._cos

    @jax.custom_jvp
    def exact_zero_cos(x):
        c = orig(x)
        return jnp.where(x == sentinel, jnp.zeros_like(c), c)

    @exact_zero_cos.defjvp
    def exact_zero_cos_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return exact_zero_cos(x), -jnp.sin(x) * t

    omega = omega.at[e[0]].set(sentinel)
    ro._cos = exact_zero_cos
    try:
        assert exact_zero_cos(omega[e[0]]) == 0.0
        g0 = jax.grad(z_q)(omega)
    finally:
        ro._cos = orig
    assert jnp.all(jnp.isfinite(g0))
    partials = _partials(a, omega, e, exact_zero_cos)
    for k, analytic in enumerate(partials):
        assert g0[e[k]] == pytest.approx(analytic, rel=1e-5, abs=1e-6)
    others0 = exact_zero_cos(omega[e[1]]) * exact_zero_cos(omega[e[2]]) * exact_zero_cos(omega[e[3]])
    assert others0 != 0
    assert g0[e[0]] == pytest.approx(-a * jnp.sin(omega[e[0]]) * others0, rel=1e-5, abs=1e-6)


def test_different_h_gives_different_z():
    n, c, d_h, d_a = 8, 3, 6, 5
    params = init_params(jax.random.PRNGKey(2), n, d_h, d_a, d_b=3)
    incident = edges(n, c).incident
    h1 = jax.random.normal(jax.random.PRNGKey(3), (d_h,))
    h2 = h1 + jnp.array([0.7, -0.2, 0.4, 0.1, -0.5, 0.3])
    a1, w1, p1 = emit(h1, params)
    a2, w2, p2 = emit(h2, params)
    assert not jnp.allclose(a1, a2)
    assert not jnp.allclose(w1, w2)
    assert not jnp.allclose(p1[0], p2[0]) or not jnp.allclose(p1[1], p2[1])
    z1 = readout(a1, w1, p1, incident)
    z2 = readout(a2, w2, p2, incident)
    assert z1.shape == (n,)
    assert not jnp.allclose(z1, z2)


def test_residual_is_scaled_linear_map():
    z = jnp.array([1.0, 2.0, 3.0])
    W_B = jnp.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    s, r = 2.0, 4.0
    out = residual(z, W_B, s, r)
    # (2/4) * [-2, 3] = [-1, 1.5]
    assert jnp.allclose(out, jnp.array([-1.0, 1.5]))
    assert jnp.allclose(out, (s / r) * (W_B @ z))
