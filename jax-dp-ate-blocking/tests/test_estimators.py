"""Exact formula checks for IPW, BPS, AIPW, and sensitivities."""

from __future__ import annotations

import jax.numpy as jnp

from dp_ate_blocking.estimators import (
    aipw,
    aipw_sensitivity,
    assign_bins,
    bps,
    clip_propensity,
    ipw,
    ipw_sensitivity,
)


def test_ipw_hand_example():
    y = jnp.array([1.0, 0.0, 2.0])
    w = jnp.array([1.0, 0.0, 1.0])
    e = jnp.array([0.5, 0.25, 0.8])
    # (1/3) * (1/0.5 - 0/(0.75) + 2/0.8) = (1/3) * (2 + 0 + 2.5) = 4.5/3
    got = float(ipw(y, w, e))
    assert abs(got - 1.5) < 1e-12


def test_ipw_two_point():
    y = jnp.array([2.0, 4.0])
    w = jnp.array([1.0, 0.0])
    e = jnp.array([0.25, 0.5])
    got = float(ipw(y, w, e))
    assert abs(got - 0.0) < 1e-12


def test_bps_hand_example_equal_spaced():
    y = jnp.array([2.0, 0.0, 4.0, 2.0])
    w = jnp.array([1.0, 0.0, 1.0, 0.0])
    e = jnp.array([0.1, 0.2, 0.6, 0.7])
    # m=2: bins [0, 0.5) and [0.5, 1]. Each bin n_k=2, n_{k,w}=1.
    # terms 4 and 4, tau = 8/4 = 2
    got = float(bps(y, w, e, m=2))
    assert abs(got - 2.0) < 1e-12


def test_bps_skips_empty_treatment_cells():
    y = jnp.array([1.0, 1.0, 0.0])
    w = jnp.array([1.0, 1.0, 0.0])
    e = jnp.array([0.1, 0.1, 0.6])
    got = float(bps(y, w, e, m=2))
    assert abs(got - 0.0) < 1e-12


def test_ipw_and_aipw_sensitivity():
    n, C_y, omega = 100.0, 1.5, 0.1
    assert abs(float(ipw_sensitivity(n, C_y, omega)) - 2.0 * C_y / (n * omega)) < 1e-15
    assert abs(float(aipw_sensitivity(n, C_y, omega)) - (4.0 * C_y / n) * (1.0 + 1.0 / omega)) < 1e-15


def test_clipping_before_ipw():
    y = jnp.array([1.0, 2.0])
    w = jnp.array([1.0, 0.0])
    e = jnp.array([0.01, 0.5])
    omega = 0.1
    got = float(ipw(y, w, e, omega=omega))
    e_clip = clip_propensity(e, omega)
    assert float(e_clip[0]) == omega
    assert float(e_clip[1]) == 0.5
    expect = float(ipw(y, w, e_clip))
    assert abs(got - expect) < 1e-15
    unclipped = float(ipw(y, w, e))
    # unclipped: 0.5 * (1/0.01 - 2/0.5) = 0.5 * (100 - 4) = 48
    # clipped:   0.5 * (1/0.1  - 2/0.5) = 0.5 * (10 - 4) = 3
    assert abs(got - 3.0) < 1e-12
    assert abs(unclipped - 48.0) < 1e-12


def test_bin_assignment_edges():
    m = 5
    e = jnp.array([0.0, 0.199999, 0.2, 0.999999, 1.0])
    bins = [int(i) for i in assign_bins(e, m)]
    assert bins[0] == 0
    assert bins[1] == 0
    assert bins[2] == 1
    assert bins[3] == m - 1
    assert bins[4] == m - 1


def test_aipw_reduces_to_ipw_when_g_zero():
    y = jnp.array([1.0, 0.0, 2.0])
    w = jnp.array([1.0, 0.0, 1.0])
    e = jnp.array([0.5, 0.25, 0.8])
    g0 = jnp.zeros_like(y)
    g1 = jnp.zeros_like(y)
    assert abs(float(aipw(y, w, e, g0, g1)) - float(ipw(y, w, e))) < 1e-15
