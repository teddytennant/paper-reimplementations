"""Checks on the constitutive laws of section 4.3 against the printed equations."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_pbe.kinetics import (
    DTDT_MAX,
    M1,
    M4,
    S_MAX,
    S_MIN,
    GrowthDissolution,
    SizeDependentGrowth,
    concentration_from_S,
    ramp_rate_penalty,
    solubility,
    supersaturation,
    supersaturation_penalty,
)

L1 = jnp.asarray(400.0)
L2 = jnp.asarray(70.0)


def test_solubility_matches_eq8():
    for T in (20.0, 28.0, 41.0):
        assert solubility(T) == pytest.approx(3.37 * np.exp(0.036 * T), rel=1e-12)


def test_supersaturation_is_one_at_solubility():
    for T in (20.0, 28.0, 41.0):
        assert supersaturation(solubility(T), T) == pytest.approx(1.0, abs=1e-12)
        assert concentration_from_S(1.0, T) == pytest.approx(solubility(T), rel=1e-12)


def test_m1_matches_published_coefficients():
    """Eqs (9) and (10): D1 = -2.32(1-S), D2 = -0.19(1-S)."""
    m = M1()
    for S in (0.975, 0.99, 1.0):
        d1, d2 = m(L1, L2, jnp.asarray(S), 28.0)
        assert float(d1) == pytest.approx(-2.32 * (1.0 - S), rel=1e-12, abs=1e-15)
        assert float(d2) == pytest.approx(-0.19 * (1.0 - S), rel=1e-12, abs=1e-15)


def test_m1_is_size_independent():
    m = M1()
    a1, a2 = m(jnp.asarray(100.0), jnp.asarray(20.0), jnp.asarray(0.98), 28.0)
    b1, b2 = m(jnp.asarray(900.0), jnp.asarray(180.0), jnp.asarray(0.98), 28.0)
    assert float(a1) == pytest.approx(float(b1), rel=1e-14)
    assert float(a2) == pytest.approx(float(b2), rel=1e-14)


def test_m4_matches_published_coefficients():
    """Eqs (11) and (12) at their published exponents."""
    m = M4()
    S = jnp.asarray(0.98)
    d1, d2 = m(L1, L2, S, 28.0)
    assert float(d1) == pytest.approx(-2.24 * 0.02 * (1.0 + 400.0) ** -5.05e-2, rel=1e-12)
    assert float(d2) == pytest.approx(-6.43e-4 * 0.02 * (1.0 + 70.0) ** 1.29, rel=1e-12)


def test_m4_size_dependence_directions():
    """The length exponent is negative and the width exponent positive, so long
    crystals dissolve slightly slower in L1 and wide ones much faster in L2."""
    m = M4()
    S = jnp.asarray(0.98)
    small = m(jnp.asarray(100.0), jnp.asarray(30.0), S, 28.0)
    large = m(jnp.asarray(800.0), jnp.asarray(150.0), S, 28.0)
    assert abs(float(large[0])) < abs(float(small[0]))
    assert abs(float(large[1])) > abs(float(small[1]))


def test_dissolution_vanishes_at_equilibrium():
    S = jnp.asarray(1.0)
    for m in (M1(), M4()):
        d1, d2 = m(L1, L2, S, 28.0)
        assert float(d1) == pytest.approx(0.0, abs=1e-14)
        assert float(d2) == pytest.approx(0.0, abs=1e-14)


def test_growth_is_zero_at_or_below_equilibrium():
    g = SizeDependentGrowth()
    for S in (0.9, 0.999, 1.0):
        g1, g2 = g(L1, L2, jnp.asarray(S), 28.0)
        assert float(g1) == 0.0
        assert float(g2) == 0.0


def test_growth_is_positive_and_gradient_is_finite_across_equilibrium():
    """The double-where in SizeDependentGrowth has to keep reverse mode clean at
    S <= 1, where (S-1)**(2/3) and ln(S)**(1/6) are otherwise nan."""
    g = SizeDependentGrowth()
    rate = lambda S: g(L1, L2, S, 28.0)[0] + g(L1, L2, S, 28.0)[1]
    grad = jax.grad(rate)
    for S in (0.90, 0.999, 1.0, 1.05, 1.25):
        v = float(rate(jnp.asarray(S)))
        d = float(grad(jnp.asarray(S)))
        assert np.isfinite(v) and np.isfinite(d)
        if S > 1.0:
            assert v > 0.0


def test_growth_length_exponent_is_1p2():
    """G1 carries L1**1.2 (eq 13); G2 carries no size dependence (eq 14)."""
    g = SizeDependentGrowth()
    S = jnp.asarray(1.1)
    a1, a2 = g(jnp.asarray(200.0), jnp.asarray(40.0), S, 30.0)
    b1, b2 = g(jnp.asarray(400.0), jnp.asarray(80.0), S, 30.0)
    assert float(b1) / float(a1) == pytest.approx(2.0**1.2, rel=1e-12)
    assert float(b2) == pytest.approx(float(a2), rel=1e-12)


def test_growth_g2_exponent_matches_eq14():
    g = SizeDependentGrowth()
    S = 1.2
    _, g2 = g(L1, L2, jnp.asarray(S), 30.0)
    expected = (S - 1.0) ** 2.7 * np.exp(-7e2 / (30.0 + 273.15))
    assert float(g2) == pytest.approx(expected, rel=1e-12)


def test_combined_model_switches_at_equilibrium():
    """Section 4.3.2: M4 below S = 1, the growth law above it."""
    c = GrowthDissolution()
    m4 = M4()
    g = SizeDependentGrowth()
    below = c(L1, L2, jnp.asarray(0.98), 28.0)
    above = c(L1, L2, jnp.asarray(1.15), 28.0)
    ref_below = m4(L1, L2, jnp.asarray(0.98), 28.0)
    ref_above = g(L1, L2, jnp.asarray(1.15), 28.0)
    assert float(below[0]) == pytest.approx(float(ref_below[0]), rel=1e-12)
    assert float(above[0]) == pytest.approx(float(ref_above[0]), rel=1e-12)
    assert float(below[0]) < 0.0 < float(above[0])


def test_supersaturation_penalty_is_zero_inside_the_validity_range():
    S = jnp.linspace(S_MIN, S_MAX, 11)
    assert float(supersaturation_penalty(S, S_MIN, S_MAX)) == 0.0
    assert float(supersaturation_penalty(jnp.asarray([0.9]), S_MIN, S_MAX)) > 0.0
    assert float(supersaturation_penalty(jnp.asarray([1.4]), S_MIN, S_MAX)) > 0.0
    # Without a ceiling only the dissolution floor is enforced.
    assert float(supersaturation_penalty(jnp.asarray([1.4]), S_MIN, None)) == 0.0


def test_ramp_rate_penalty_matches_the_vessel_limit():
    """|dT/dt| <= 0.05 C/min, section 4.6."""
    t = jnp.linspace(0.0, 3600.0, 61)
    ok = t * (DTDT_MAX * 0.9) + 25.0
    too_fast = t * (DTDT_MAX * 2.0) + 25.0
    assert float(ramp_rate_penalty(ok, t)) == 0.0
    # The excess is relative to the limit, so twice the limit costs (2 - 1)**2.
    assert float(ramp_rate_penalty(too_fast, t)) == pytest.approx(1.0, rel=1e-10)
