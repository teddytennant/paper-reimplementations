"""Reverse mode gradients through the solver against central finite differences.

This is the property the whole paper rests on: the trajectory loss has to be
differentiable with respect to the kinetic parameters, the initial distribution
and the temperature profile, all at once. Everything runs in float64, so the
finite difference reference is good to about 1e-9 relative and the tolerances
below are set well inside that.
"""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import pytest

from hybrid_pbe.data import lognormal_seed
from hybrid_pbe.grid import Grid, aspect_ratio
from hybrid_pbe.kinetics import M1, M4, GrowthDissolution, concentration_from_S
from hybrid_pbe.solver import TemperatureProfile, simulate

GRID = Grid(n1=24, n2=12, L1_max=1000.0, L2_max=200.0)
T_GRID = jnp.linspace(0.0, 1800.0, 7)
SEED = lognormal_seed(GRID, 350.0, 0.30, 60.0, 0.25, 1.0e12)
C0 = concentration_from_S(0.98, 29.0)


def central_difference(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def _trajectory_loss(kinetics, f0, T_values, c0):
    profile = TemperatureProfile(times=T_GRID, values=T_values)
    res = simulate(f0, c0, profile, T_GRID, kinetics, GRID, n_substeps=4)
    return jnp.sum(res.c**2) + jnp.sum(res.L1**2) * 1e-4 + jnp.sum(res.L2**2) * 1e-3


T_RAMP = jnp.linspace(29.0, 31.0, 7)


def test_gradient_wrt_kinetic_parameter():
    """d(loss)/dk1 for the M1 coefficient of eq (9)."""
    fn = lambda k1: _trajectory_loss(M1(k1=k1, k2=0.19), SEED, T_RAMP, C0)
    analytic = float(jax.grad(fn)(2.32))
    numeric = float(central_difference(fn, 2.32, 1e-5))
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_gradient_wrt_all_four_m4_parameters():
    """M4 (eqs 11 and 12) has a size dependent exponent inside a power, which is
    where a broken custom derivative would show up."""
    base = jnp.asarray([2.24, -5.05e-2, 6.43e-4, 1.29])

    def fn(p):
        return _trajectory_loss(M4(p[0], p[1], p[2], p[3]), SEED, T_RAMP, C0)

    analytic = jax.grad(fn)(base)
    for i, h in enumerate([1e-6, 1e-8, 1e-9, 1e-6]):
        step = jnp.zeros(4).at[i].set(h)
        numeric = (fn(base + step) - fn(base - step)) / (2.0 * h)
        assert float(analytic[i]) == pytest.approx(float(numeric), rel=1e-5)


def test_gradient_wrt_initial_distribution():
    """f0 is the largest block of free variables in Case B, section 4.5."""
    cells = [(8, 4), (12, 6), (16, 3)]

    def fn(f0):
        return _trajectory_loss(M1(), f0, T_RAMP, C0)

    analytic = jax.grad(fn)(SEED)
    scale = float(jnp.max(SEED))
    for i, j in cells:
        h = 1e-4 * scale
        step = jnp.zeros_like(SEED).at[i, j].set(h)
        numeric = (fn(SEED + step) - fn(SEED - step)) / (2.0 * h)
        assert float(analytic[i, j]) == pytest.approx(float(numeric), rel=1e-6)


def test_gradient_wrt_temperature_profile():
    """T(t) is the free variable for the process optimisation of section 4.6."""

    def fn(T_values):
        return _trajectory_loss(M1(), SEED, T_values, C0)

    analytic = jax.grad(fn)(T_RAMP)
    for i in (0, 3, 6):
        h = 1e-6
        step = jnp.zeros_like(T_RAMP).at[i].set(h)
        numeric = (fn(T_RAMP + step) - fn(T_RAMP - step)) / (2.0 * h)
        assert float(analytic[i]) == pytest.approx(float(numeric), rel=1e-6)


def test_gradient_wrt_initial_concentration():
    fn = lambda c0: _trajectory_loss(M1(), SEED, T_RAMP, c0)
    analytic = float(jax.grad(fn)(C0))
    numeric = float(central_difference(fn, C0, 1e-7))
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_gradient_of_aspect_ratio_through_the_growth_dissolution_model():
    """The objective of eq (22) with the combined law, which contains the
    where-guarded growth branch. A nan in the guard shows up here first."""
    t = jnp.linspace(0.0, 1200.0, 7)
    T_values = jnp.linspace(32.0, 28.0, 7)
    c0 = concentration_from_S(0.98, 32.0)

    def fn(T_values):
        profile = TemperatureProfile(times=t, values=T_values)
        res = simulate(SEED, c0, profile, t, GrowthDissolution(), GRID, n_substeps=4)
        return aspect_ratio(res.f_final, GRID)

    analytic = jax.grad(fn)(T_values)
    assert bool(jnp.all(jnp.isfinite(analytic)))
    assert float(jnp.max(jnp.abs(analytic))) > 0.0
    for i in (0, 4):
        h = 1e-6
        step = jnp.zeros_like(T_values).at[i].set(h)
        numeric = (fn(T_values + step) - fn(T_values - step)) / (2.0 * h)
        assert float(analytic[i]) == pytest.approx(float(numeric), rel=1e-5, abs=1e-9)


def test_gradient_wrt_neural_rate_weights():
    """The whole point of section 4.4: gradients reach the MLP weights through
    every solver step."""
    from hybrid_pbe.hybrid import NeuralDissolution

    key = jax.random.PRNGKey(0)
    net = NeuralDissolution(key, s_mean=0.985, s_std=0.01, t_mean=30.0, t_std=2.0,
                            ref_rate=jnp.asarray([0.05, 0.005]), width=6, depth=2)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def fn(params):
        return _trajectory_loss(eqx.combine(params, static), SEED, T_RAMP, C0)

    analytic = jax.grad(fn)(params)
    flat = jax.tree_util.tree_leaves(analytic)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in flat)
    assert max(float(jnp.max(jnp.abs(g))) for g in flat) > 0.0

    # Check one weight against a central difference.
    w = params.mlp.layers[0].weight
    h = 1e-6
    bumped_up = eqx.tree_at(lambda p: p.mlp.layers[0].weight, params,
                            w.at[0, 0].add(h))
    bumped_dn = eqx.tree_at(lambda p: p.mlp.layers[0].weight, params,
                            w.at[0, 0].add(-h))
    numeric = (fn(bumped_up) - fn(bumped_dn)) / (2.0 * h)
    assert float(analytic.mlp.layers[0].weight[0, 0]) == pytest.approx(
        float(numeric), rel=1e-6)
