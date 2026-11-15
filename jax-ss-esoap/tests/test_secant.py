"""Theorem 4.2, the directional secant matching that motivates equation (3)."""

import jax.numpy as jnp
import numpy as np
import pytest

import quadratic as qd
from ss_esoap._core import eigh_basis, kron_quadratic_form, secant_scale


def _setup(seed, factor_scale):
    """A step pair ``(S, Y)`` on a quadratic, plus a Kronecker metric ``(L, R)``.

    ``Y = H S`` holds exactly here, so the secant residual ``r`` of Theorem 4.2 is
    zero and the theorem predicts ``s^T (tau K) s = s^T H s`` on the nose.
    ``factor_scale`` moves ``a_k`` around, inversely, since ``L`` and ``R`` enter
    ``a_k`` inverted, so both branches of the ``min(1, .)`` get exercised.
    """
    rng = np.random.default_rng(seed)
    a, b, c, x0, _ = qd.make(seed)
    s = jnp.asarray(rng.standard_normal((qd.M, qd.N)))
    # y = grad f(x0 + s) - grad f(x0) = A^T A S B B^T for a quadratic.
    g0 = a.T @ (a @ x0 @ b - c) @ b.T
    g1 = a.T @ (a @ (x0 + s) @ b - c) @ b.T
    y = g1 - g0
    ell = factor_scale * jnp.asarray(_spd(rng, qd.M))
    r = factor_scale * jnp.asarray(_spd(rng, qd.N))
    return s, y, ell, r, a, b


def _spd(rng, d):
    m = rng.standard_normal((d, d))
    return m @ m.T / d + np.eye(d)


def _tau_and_energy(s, y, ell, r, tau_min=0.0):
    ql, qr = eigh_basis(ell), eigh_basis(r)
    dl = jnp.diagonal(ql.T @ ell @ ql)
    dr = jnp.diagonal(qr.T @ r @ qr)
    tau = secant_scale(s, y, ql, qr, dl, dr, tau_min, 0.0)
    return tau, kron_quadratic_form(s, ell, r)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eigenspace_shortcut_matches_dense_quadratic_form(seed):
    """Equation (5): the eigenbasis sum equals ``Tr(S^T L^-1 S R^-1)``."""
    s, y, ell, r, _, _ = _setup(seed, 1.0)
    ql, qr = eigh_basis(ell), eigh_basis(r)
    st = ql.T @ s @ qr
    dl = jnp.diagonal(ql.T @ ell @ ql)
    dr = jnp.diagonal(qr.T @ r @ qr)
    a_eig = jnp.sum(st**2 / (dl[:, None] * dr[None, :]))
    assert jnp.allclose(a_eig, kron_quadratic_form(s, ell, r), rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_scaled_kronecker_energy_equals_directional_curvature(seed):
    """Theorem 4.2 with ``r_k = 0``: ``s^T (tau K) s = s^T H s`` when unclipped.

    The scale on ``(L, R)`` is picked so the unclipped branch is the active one,
    which is the branch the theorem's identity is stated for.
    """
    s, y, ell, r, a, b = _setup(seed, 1.0)
    tau, energy = _tau_and_energy(s, y, ell, r)
    assert 0.0 < float(tau) < 1.0, "test needs the unclipped branch"

    h = qd.dense_hessian(a, b)
    sv = qd.vec(s)
    true_curv = sv @ (h @ sv)
    assert jnp.allclose(tau * energy, true_curv, rtol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mismatch_is_exactly_the_directional_secant_error(seed):
    """Theorem 4.2 with ``y = H s + r``: the gap is exactly ``s^T r``."""
    rng = np.random.default_rng(100 + seed)
    s, y, ell, r_fac, a, b = _setup(seed, 1.0)
    resid = jnp.asarray(0.05 * rng.standard_normal(y.shape) * float(jnp.std(y)))
    tau, energy = _tau_and_energy(s, y + resid, ell, r_fac)
    assert 0.0 < float(tau) < 1.0

    h = qd.dense_hessian(a, b)
    sv = qd.vec(s)
    gap = tau * energy - sv @ (h @ sv)
    assert jnp.allclose(gap, jnp.sum(s * resid), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tau_minimizes_the_clipped_log_secant_objective(seed):
    """Theorem 4.2's variational statement, checked by brute force.

    ``tau`` should minimize ``J(tau) = [log(c / (tau a))]^2`` over ``0 < tau <= 1``.
    """
    s, y, ell, r, _, _ = _setup(seed, 1.0)
    tau, a_k = _tau_and_energy(s, y, ell, r)
    c_k = jnp.sum(y * s)

    def j(t):
        return jnp.log(c_k / (t * a_k)) ** 2

    grid = jnp.geomspace(1e-4, 1.0, 40001)
    best = grid[jnp.argmin(j(grid))]
    assert float(j(tau)) <= float(j(best)) + 1e-12
    assert jnp.allclose(tau, best, rtol=2e-3)


def test_tau_clips_to_one_when_the_metric_is_soft():
    """Equation (3)'s ``min(1, .)``: a metric that underestimates curvature gives
    ``c_k / a_k > 1``, and the clip selects ``tau = 1``."""
    s, y, ell, r, _, _ = _setup(0, 1e3)
    tau, a_k = _tau_and_energy(s, y, ell, r)
    assert float(jnp.sum(y * s) / a_k) > 1.0
    assert float(tau) == pytest.approx(1.0)


def test_tau_falls_back_to_one_on_negative_curvature():
    """Algorithm 2 step 4: ``c_k <= 0`` means no usable secant, so ``tau = 1``."""
    s, y, ell, r, _, _ = _setup(0, 1.0)
    tau, _ = _tau_and_energy(s, -y, ell, r)
    assert float(tau) == pytest.approx(1.0)


def test_tau_min_floors_the_step_amplification():
    """``tau^{-1/2}`` is the step multiplier, so ``tau_min`` caps it."""
    s, y, ell, r, _, _ = _setup(0, 1.0)
    tau, _ = _tau_and_energy(s, y, ell, r, tau_min=0.1)
    assert float(tau) == pytest.approx(0.1)
