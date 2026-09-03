"""Equations 12-15."""

import jax.numpy as jnp
import pytest

from assert_sampler import schedules


def test_cosine_decay_endpoints_and_monotonicity():
    # Equation 13: D(0) = 1, D(1) = 0, D(1/2) = 1/2.
    p = jnp.linspace(0.0, 1.0, 101)
    d = schedules.cosine_decay(p)
    assert float(d[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(d[-1]) == pytest.approx(0.0, abs=1e-6)
    assert float(schedules.cosine_decay(jnp.array(0.5))) == pytest.approx(0.5, abs=1e-6)
    assert bool(jnp.all(jnp.diff(d) <= 1e-7))


@pytest.mark.parametrize("num_steps", [2, 10, 50, 137])
def test_assert_schedule_endpoints(num_steps):
    eta = schedules.assert_schedule(num_steps, eta_max=0.8, tau_trans=0.1)
    # Section 3.3: i = 0 is the early high-noise step and dividing by S - 1 forces
    # eta_{S-1} = 0.
    assert float(eta[0]) == pytest.approx(0.8, abs=1e-6)
    assert float(eta[-1]) == pytest.approx(0.0, abs=1e-7)
    assert bool(jnp.all(jnp.diff(eta) <= 1e-7))
    assert bool(jnp.all((eta >= 0.0) & (eta <= 0.8 + 1e-7)))


def test_assert_schedule_matches_equation_12_termwise():
    s, eta_max, tau = 40, 0.8, 0.1
    eta = schedules.assert_schedule(s, eta_max=eta_max, tau_trans=tau)
    for i in range(s):
        r = i / (s - 1)
        if r < tau:
            want = eta_max
        else:
            want = eta_max * 0.5 * (1.0 + jnp.cos(jnp.pi * (r - tau) / (1.0 - tau)))
        assert float(eta[i]) == pytest.approx(float(want), abs=1e-6)


def test_plateau_length_follows_tau_trans():
    s = 100
    for tau in [0.0, 0.1, 0.4]:
        eta = schedules.assert_schedule(s, eta_max=0.8, tau_trans=tau)
        held = int(jnp.sum(jnp.isclose(eta, 0.8, atol=1e-7)))
        expected = int(jnp.sum(schedules.progress(s) < tau))
        # r = tau itself lands exactly on D(0) = 1, so it is also at eta_max.
        assert held in (expected, expected + 1)


def test_tau_zero_is_pure_cosine():
    s = 30
    eta = schedules.assert_schedule(s, eta_max=1.0, tau_trans=0.0)
    assert jnp.allclose(eta, schedules.cosine_decay(schedules.progress(s)), atol=1e-6)


def test_baseline_schedules_match_equations_14_and_15():
    s, eta_max, eta_min, tau = 25, 0.8, 0.0, 0.4
    r = schedules.progress(s)
    lin = schedules.linear_decay_schedule(s, eta_max, eta_min)
    assert jnp.allclose(lin, eta_max - r * (eta_max - eta_min), atol=1e-7)
    stp = schedules.step_schedule(s, eta_max, eta_min, tau)
    rev = schedules.reverse_step_schedule(s, eta_max, eta_min, tau)
    assert jnp.allclose(stp, jnp.where(r < tau, eta_max, eta_min))
    assert jnp.allclose(rev, jnp.where(r < tau, eta_min, eta_max))
    # The two step schedules are exact mirrors of one another.
    assert jnp.allclose(stp + rev, eta_max + eta_min)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        schedules.assert_schedule(10, tau_trans=1.0)
    with pytest.raises(ValueError):
        schedules.progress(1)
