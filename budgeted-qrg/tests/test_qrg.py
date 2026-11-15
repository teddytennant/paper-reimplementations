"""Lock displayed equations from arXiv:2609.31222, not a rewrite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrg.kernel import (
    LOCAL,
    PREDNEXT,
    budgeted_delivery,
    budgeted_descent,
    combine_step,
    covector_lift,
    evaluation_state,
    horizontal_lift,
    product_budget_delivery,
    qrg_step,
    sampler_budget,
)
from qrg.quotients import pairwise_contact, pairwise_distance


def _two_atoms(sep=3.0):
    return jnp.array([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])


def _rot_z(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_horizontal_lift_reproduces_delta_q_and_row_space():
    x = _two_atoms(2.5)
    _, J = pairwise_distance(x, 0, 1)
    delta_q = jnp.array([0.4])
    dx = horizontal_lift(J, delta_q)
    # J full row rank: J Δx = Δq
    np.testing.assert_allclose(J @ dx, delta_q, rtol=1e-6, atol=1e-6)
    # Δx lies in the row space of J (range of J^T)
    proj = J.T @ jnp.linalg.pinv(J @ J.T) @ (J @ dx)
    np.testing.assert_allclose(dx, proj, rtol=1e-6, atol=1e-6)
    # matches the displayed right-inverse formula
    gram = J @ J.T
    explicit = J.T @ jnp.linalg.solve(gram, delta_q)
    np.testing.assert_allclose(dx, explicit, rtol=1e-6, atol=1e-6)


def test_rank_deficient_uses_stable_pinv_not_skip():
    x = _two_atoms(2.0)
    _, J_d = pairwise_distance(x, 0, 1)
    J = jnp.concatenate([J_d, 2.0 * J_d], axis=0)
    assert jnp.linalg.matrix_rank(J) == 1
    # consistent residual: least-squares / min-norm lift is nonzero
    delta_q = jnp.array([0.5, 1.0])
    dx = horizontal_lift(J, delta_q)
    assert jnp.linalg.norm(dx) > 1e-6
    np.testing.assert_allclose(J @ dx, delta_q, rtol=1e-5, atol=1e-5)
    # inconsistent residual still returns a lift (Gram pinv), not a skip
    dx_inc = horizontal_lift(J, jnp.array([0.5, 0.0]))
    assert jnp.linalg.norm(dx_inc) > 1e-6
    # still horizontal
    proj = J.T @ jnp.linalg.pinv(J @ J.T) @ (J @ dx_inc)
    np.testing.assert_allclose(dx_inc, proj, rtol=1e-5, atol=1e-5)


def test_covector_lift_eq3_euclidean():
    x = _two_atoms(1.5)
    _, J = pairwise_distance(x, 0, 1)
    c = jnp.array([0.3])
    h = covector_lift(J, c)
    np.testing.assert_allclose(h, J.T @ c, rtol=1e-6, atol=1e-6)
    # Euclidean-horizontal: h in range(J^T)
    proj = J.T @ jnp.linalg.pinv(J @ J.T) @ (J @ h)
    np.testing.assert_allclose(h, proj, rtol=1e-6, atol=1e-6)


def test_budget_clamp_matches_eq8_scale():
    dx_q = jnp.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dx_s = jnp.array([0.5, 0.0, 0.0, -0.5, 0.0, 0.0])
    rho = 1.0
    nq = float(jnp.linalg.norm(dx_q))
    ns = float(jnp.linalg.norm(dx_s))
    assert nq > rho * ns
    delivered = budgeted_delivery(dx_q, dx_s, rho=rho)
    np.testing.assert_allclose(
        jnp.linalg.norm(delivered), rho * ns, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        delivered / jnp.linalg.norm(delivered),
        dx_q / nq,
        rtol=1e-6,
        atol=1e-6,
    )
    # eq (5): B = ρ ||v||
    np.testing.assert_allclose(sampler_budget(dx_s, rho=rho), rho * ns)
    # eq (8) inserts the minus on a covector lift
    u = budgeted_descent(dx_q, dx_s, rho=rho)
    np.testing.assert_allclose(u, -delivered, rtol=1e-6, atol=1e-6)


def test_budget_noop_when_delta_x_q_is_zero():
    dx_s = jnp.array([0.2, -0.1, 0.0, 0.0, 0.3, 0.0])
    out = budgeted_delivery(jnp.zeros_like(dx_s), dx_s, rho=1.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_combine_step_adds_sampler_and_qrg():
    x = _two_atoms(1.0)
    v = jnp.ones_like(x) * 0.1
    u = -jnp.ones_like(x) * 0.05
    np.testing.assert_allclose(combine_step(x, v, u), x + v + u)


def test_prednext_uses_predicted_next_state_local_uses_current():
    x = _two_atoms(3.0)
    v = jnp.array([[0.0, 0.0, 0.0], [-0.8, 0.0, 0.0]])
    target = jnp.array(2.0)
    q_now, _ = pairwise_distance(x, 0, 1)
    q_next, _ = pairwise_distance(x + v, 0, 1)
    assert abs(float(q_now - q_next)) > 0.2

    np.testing.assert_allclose(evaluation_state(x, v, LOCAL), x)
    np.testing.assert_allclose(evaluation_state(x, v, PREDNEXT), x + v)

    x_local = qrg_step(x, v, lambda z: pairwise_distance(z, 0, 1), target, mode=LOCAL)
    x_pred = qrg_step(x, v, lambda z: pairwise_distance(z, 0, 1), target, mode=PREDNEXT)
    # Different evaluation states ⇒ different residual lifts
    assert jnp.linalg.norm(x_local - x_pred) > 1e-4

    def expected(x_eval):
        q, J = pairwise_distance(x_eval, 0, 1)
        dx_q = horizontal_lift(J, target - q).reshape(x.shape)
        return combine_step(x, v, budgeted_delivery(dx_q, v, rho=1.0))

    np.testing.assert_allclose(x_local, expected(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x_pred, expected(x + v), rtol=1e-5, atol=1e-5)


def test_product_budget_section_plus_residual_cannot_exceed_sampler():
    v = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sec = 8.0 * v
    res = 8.0 * v
    ns = float(jnp.linalg.norm(v))
    u = product_budget_delivery(sec, res, v, rho_s=1.0, rho_r=1.0, rho=1.0)
    assert float(jnp.linalg.norm(u)) <= ns + 1e-5
    # Independent eq-(14) caps would allow up to 2 ||v||; remaining budget forbids it
    u_sec_only = budgeted_delivery(sec, v, rho=1.0)
    np.testing.assert_allclose(jnp.linalg.norm(u_sec_only), ns, rtol=1e-6)
    np.testing.assert_allclose(u, u_sec_only, rtol=1e-5, atol=1e-5)
    # Residual-only still spends its own budget
    u_res = product_budget_delivery(
        jnp.zeros_like(res), res, v, rho_s=1.0, rho_r=1.0, rho=1.0
    )
    np.testing.assert_allclose(jnp.linalg.norm(u_res), ns, rtol=1e-6)


def test_distance_quotient_se3_equivariance():
    x = jnp.array([[0.2, -0.4, 0.1], [1.5, 0.3, -0.7]])
    q, J = pairwise_distance(x, 0, 1)
    delta_q = jnp.array([0.25])
    dx = horizontal_lift(J, delta_q).reshape(x.shape)

    R = _rot_z(0.9)
    x_rot = x @ R.T
    q_rot, J_rot = pairwise_distance(x_rot, 0, 1)
    np.testing.assert_allclose(q_rot, q, rtol=1e-5, atol=1e-5)
    dx_rot = horizontal_lift(J_rot, delta_q).reshape(x.shape)
    np.testing.assert_allclose(dx_rot, dx @ R.T, rtol=1e-5, atol=1e-5)

    t = jnp.array([4.0, -2.0, 1.5])
    x_tr = x + t
    q_tr, J_tr = pairwise_distance(x_tr, 0, 1)
    np.testing.assert_allclose(q_tr, q, rtol=1e-5, atol=1e-5)
    dx_tr = horizontal_lift(J_tr, delta_q).reshape(x.shape)
    np.testing.assert_allclose(dx_tr, dx, rtol=1e-5, atol=1e-5)

    v = jnp.array([[0.05, -0.02, 0.01], [-0.04, 0.03, 0.0]])
    u = budgeted_delivery(dx, v, rho=1.0)
    u_rot = budgeted_delivery(dx_rot, v @ R.T, rho=1.0)
    np.testing.assert_allclose(u_rot, u @ R.T, rtol=1e-5, atol=1e-5)


def test_contact_jacobian_matches_autodiff():
    x = jnp.array([[0.0, 0.0, 0.0], [1.2, 0.4, -0.3], [2.0, 1.0, 0.0]])
    q, J = pairwise_contact(x, 0, 2, cutoff=2.0, steepness=1.5)

    def q_flat(z):
        return pairwise_contact(z.reshape(3, 3), 0, 2, cutoff=2.0, steepness=1.5)[0]

    J_ad = jax.jacrev(q_flat)(x.reshape(-1))
    np.testing.assert_allclose(J.reshape(-1), J_ad, rtol=1e-5, atol=1e-5)
    assert 0.0 < float(q) < 1.0


def test_qrg_step_unknown_mode_raises():
    x = _two_atoms()
    with pytest.raises(ValueError):
        evaluation_state(x, jnp.zeros_like(x), mode="rollout")
