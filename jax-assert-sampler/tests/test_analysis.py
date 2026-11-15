"""Equations 8-11: the first-order error recursion, its unrolling, and the
decorrelation claim."""

import jax
import jax.numpy as jnp
import pytest

from assert_sampler import analysis, cim, schedules, toy


STEPS = 20
BATCH = 128
LEVEL = 0.03


@pytest.fixture(scope="module")
def case():
    t = toy.setup(n_samples=BATCH)
    noise = cim.calibrate(t.params, LEVEL)
    fixed = analysis.as_param_tangent(
        t.params, cim.sample_mapping(jax.random.key(100), t.params, noise))
    per_step = [analysis.as_param_tangent(
        t.params, cim.sample_mapping(jax.random.key(500 + i), t.params, noise))
        for i in range(STEPS)]
    return t, fixed, per_step


def _traj(case, etas):
    t, _, _ = case
    return analysis.reference_trajectory(jax.random.key(3), t.params, t.sched,
                                         STEPS, etas, t.x_init)


DET = schedules.constant_schedule(STEPS, 0.0)
ASSERT = schedules.assert_schedule(STEPS)


@pytest.fixture(scope="module")
def terms(case):
    """u_i for every (eta schedule, Delta theta) combination, computed once.

    Each entry is Equation 9's propagated per-step contribution, stacked over i.
    """
    t, fixed, per_step = case
    out = {}
    for eta_name, etas in [("det", DET), ("assert", ASSERT)]:
        xs, zs, step_fn = _traj(case, etas)
        for map_name, dtheta in [("fixed", fixed), ("resampled", per_step)]:
            out[(eta_name, map_name)] = analysis.propagated_terms(
                t.params, dtheta, xs, zs, step_fn)
    return out


def test_linearization_matches_finite_differences(case):
    """Equation 8 against (x_S(theta + h dtheta) - x_S(theta)) / h on the real
    nonlinear sampler, with the z_i shared between the two runs."""
    t, fixed, _ = case
    xs, zs, step_fn = _traj(case, DET)
    e = analysis.linearized_error(t.params, fixed, xs, zs, step_fn)
    errs = []
    for h in [1e-1, 1e-2, 1e-3]:
        fd = analysis.finite_difference_error(t.params, fixed, xs, zs, step_fn, h)
        errs.append(float(jnp.linalg.norm(fd - e) / jnp.linalg.norm(e)))
    # First order means the residual shrinks with h, and is small once h is.
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.03
    assert float(jnp.linalg.norm(e)) > 0.0


def test_linearization_matches_finite_differences_on_a_stochastic_trajectory(case):
    t, fixed, _ = case
    xs, zs, step_fn = _traj(case, ASSERT)
    e = analysis.linearized_error(t.params, fixed, xs, zs, step_fn)
    fd = analysis.finite_difference_error(t.params, fixed, xs, zs, step_fn, 1e-3)
    assert float(jnp.linalg.norm(fd - e) / jnp.linalg.norm(e)) < 0.03


def test_unrolled_sum_equals_the_recursion(case, terms):
    """Equation 9: e_S = sum_i P_{S,i+1} d_i."""
    t, fixed, _ = case
    xs, zs, step_fn = _traj(case, ASSERT)
    us = terms[("assert", "fixed")]
    e = analysis.linearized_error(t.params, fixed, xs, zs, step_fn)
    assert us.shape == (STEPS,) + xs[0].shape
    assert float(jnp.linalg.norm(us.sum(0) - e) / jnp.linalg.norm(e)) < 1e-4


def test_decomposition_is_an_identity(terms):
    """Equation 10 is algebraic, so marginal + cross must equal ||sum u_i||^2."""
    us = terms[("det", "fixed")]
    marginal, cross, total = analysis.decompose(us)
    assert float(marginal + cross) == pytest.approx(float(total), rel=1e-5)
    explicit = 2.0 * sum(float(jnp.sum(us[i] * us[j]))
                         for i in range(STEPS) for j in range(i + 1, STEPS))
    assert float(cross) == pytest.approx(explicit, rel=1e-3)


def test_inactive_steps_contribute_nothing(case):
    """a_i = 0 for all i gives e_S = 0 exactly."""
    t, fixed, _ = case
    xs, zs, step_fn = _traj(case, DET)
    e = analysis.linearized_error(t.params, fixed, xs, zs, step_fn,
                                  active=jnp.zeros(STEPS))
    assert float(jnp.max(jnp.abs(e))) == 0.0


def test_persistent_noise_makes_the_cross_terms_dominate(terms):
    """The premise of Section 3.2. One reused Delta theta leaves the propagated
    per-step contributions strongly aligned, so the cross terms of Equation 10 carry
    most of ||e_S||^2. Independent per-step Delta theta_i is the control."""
    us_fixed = terms[("det", "fixed")]
    us_free = terms[("det", "resampled")]
    assert float(analysis.cross_step_cosine(us_fixed)) > 0.35
    assert abs(float(analysis.cross_step_cosine(us_free))) < 0.08
    m_f, c_f, _ = (float(v) for v in analysis.decompose(us_fixed))
    m_g, c_g, _ = (float(v) for v in analysis.decompose(us_free))
    assert c_f / m_f > 3.0
    assert abs(c_g) / m_g < 1.5


def _cosines(terms, map_name):
    return {eta: float(analysis.cross_step_cosine(terms[(eta, map_name)]))
            for eta in ("det", "assert")}


def test_assert_reduces_cross_step_alignment(terms):
    """Equation 11, and the sentence in the abstract it formalizes. Injected early
    stochasticity changes x_{t_i}, hence H_i, hence the direction of d_i for the same
    physical Delta theta, so the propagated contributions stop pointing the same way."""
    cos = _cosines(terms, "fixed")
    assert cos["det"] > 0.35
    assert cos["assert"] < 0.9 * cos["det"]
    assert cos["det"] - cos["assert"] > 0.08


def test_stochasticity_has_nothing_to_decorrelate_once_the_map_is_resampled(terms):
    """The control. With an independent Delta theta_i per step the propagated
    contributions are already unaligned, so ASSERT cannot reduce an alignment that is
    not there, and the drop it produces collapses next to the fixed-map case."""
    fixed_cos = _cosines(terms, "fixed")
    fixed_drop = fixed_cos["det"] - fixed_cos["assert"]
    free = _cosines(terms, "resampled")
    free_drop = free["det"] - free["assert"]
    assert abs(free["det"]) < 0.08 and abs(free["assert"]) < 0.08
    assert abs(free_drop) < 0.04
    assert fixed_drop > 3.0 * abs(free_drop)


def test_early_steps_carry_more_first_order_error_than_late_steps(case):
    """The Figure 4 ordering, read off Equation 9 instead of off FID: equal-length
    early and late activation windows, same Delta theta, same trajectory."""
    t, fixed, _ = case
    xs, zs, step_fn = _traj(case, DET)
    w = 5
    early = jnp.concatenate([jnp.ones(w), jnp.zeros(STEPS - w)])
    late = jnp.concatenate([jnp.zeros(STEPS - w), jnp.ones(w)])
    e_early = analysis.linearized_error(t.params, fixed, xs, zs, step_fn, active=early)
    e_late = analysis.linearized_error(t.params, fixed, xs, zs, step_fn, active=late)
    n_early = float(jnp.linalg.norm(e_early))
    n_late = float(jnp.linalg.norm(e_late))
    assert n_early > 3.0 * n_late


def test_per_step_tangent_length_is_checked(case):
    t, fixed, per_step = case
    xs, zs, step_fn = _traj(case, DET)
    with pytest.raises(ValueError):
        analysis.linearized_error(t.params, per_step[:3], xs, zs, step_fn)
