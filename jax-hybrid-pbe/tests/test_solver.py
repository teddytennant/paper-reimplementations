"""Properties the finite volume discretisation of eqs (1) to (6) has to satisfy.

These are not smoke tests. Each one checks a closed form result: exact advection
at Courant 1, the method of moments identities dmu_k/dt = k G mu_(k-1) including
the size of the discretisation error, first order convergence to f0(L - Gt), the
liquid/solid mass balance of eq (2), and positivity under the CFL condition.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_pbe.data import lognormal_seed
from hybrid_pbe.grid import Grid, crystal_volume, moment
from hybrid_pbe.kinetics import (
    M1,
    M4,
    ConstantRate,
    GrowthDissolution,
    concentration_from_S,
    solubility,
)
from hybrid_pbe.solver import (
    ConstantTemperature,
    TemperatureProfile,
    simulate,
    substeps_for_cfl,
    total_mass,
)

T_FIXED = ConstantTemperature(28.0)


def gaussian(grid, mu1, s1, mu2, s2, amp=1.0):
    """Un-normalised bivariate Gaussian, so a shifted copy is the exact solution."""
    l1 = grid.L1_centers[:, None]
    l2 = grid.L2_centers[None, :]
    return amp * jnp.exp(-0.5 * (((l1 - mu1) / s1) ** 2 + ((l2 - mu2) / s2) ** 2))


def test_upwind_is_exact_translation_at_courant_one_along_L1():
    """Donor cell advection at Courant exactly 1 in one direction shifts the cell
    averages by whole cells, so the answer is f0(L1 - G t) with no error at all."""
    grid = Grid(n1=48, n2=24, L1_max=480.0, L2_max=240.0)
    f0 = gaussian(grid, 150.0, 30.0, 100.0, 25.0, amp=100.0)
    u1, n_shift = 0.02, 5
    dt = grid.dL1 / u1
    t = jnp.asarray([0.0, dt * n_shift])
    res = simulate(f0, 0.0, T_FIXED, t, ConstantRate(u1, 0.0), grid,
                   n_substeps=n_shift, couple=False)
    shifted = jnp.concatenate([jnp.zeros((n_shift, grid.n2)), f0[:-n_shift]], axis=0)
    assert float(res.courant_max) == pytest.approx(1.0, rel=1e-12)
    assert float(jnp.max(jnp.abs(res.f_final - shifted))) < 1e-12


def test_upwind_is_exact_translation_at_courant_one_along_L2():
    grid = Grid(n1=24, n2=48, L1_max=240.0, L2_max=480.0)
    f0 = gaussian(grid, 100.0, 25.0, 150.0, 30.0, amp=100.0)
    u2, n_shift = 0.01, 4
    dt = grid.dL2 / u2
    t = jnp.asarray([0.0, dt * n_shift])
    res = simulate(f0, 0.0, T_FIXED, t, ConstantRate(0.0, u2), grid,
                   n_substeps=n_shift, couple=False)
    shifted = jnp.concatenate([jnp.zeros((grid.n1, n_shift)), f0[:, :-n_shift]], axis=1)
    assert float(jnp.max(jnp.abs(res.f_final - shifted))) < 1e-12


def _translation_error(n, cfl=0.5):
    grid = Grid(n1=n, n2=n // 2, L1_max=480.0, L2_max=240.0)
    f0 = gaussian(grid, 150.0, 40.0, 100.0, 30.0)
    u1, u2, t_end = 0.02, 0.01, 3000.0
    n_sub = substeps_for_cfl(grid, t_end, u1, u2, cfl=cfl)
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, t_end]),
                   ConstantRate(u1, u2), grid, n_substeps=n_sub, couple=False)
    exact = gaussian(grid, 150.0 + u1 * t_end, 40.0, 100.0 + u2 * t_end, 30.0)
    assert float(res.courant_max) <= cfl + 1e-12
    return float(jnp.sum(jnp.abs(res.f_final - exact)) / jnp.sum(exact))


def test_translation_converges_first_order_to_the_analytic_solution():
    """Away from Courant 1 the scheme is diffusive, so the check is that the L1
    error against f0(L - Gt) falls at the first order rate under refinement."""
    coarse = _translation_error(32)
    fine = _translation_error(64)
    assert coarse < 0.30
    assert fine < 0.15
    assert coarse / fine > 1.5


def test_zeroth_moment_is_conserved_under_pure_growth():
    """No nucleation and nothing reaching a boundary, so mu_00 is constant."""
    grid = Grid(n1=64, n2=32, L1_max=640.0, L2_max=320.0)
    f0 = gaussian(grid, 150.0, 30.0, 90.0, 20.0, amp=100.0)
    t_end = 3000.0
    kin = ConstantRate(0.02, 0.01)
    n_sub = substeps_for_cfl(grid, t_end, 0.02, 0.01, cfl=0.5)
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, t_end]), kin, grid,
                   n_substeps=n_sub, couple=False)
    m0 = moment(f0, grid, 0, 0)
    m1 = moment(res.f_final, grid, 0, 0)
    assert float(abs(m1 - m0) / m0) < 1e-12


def test_first_moments_follow_the_method_of_moments_exactly():
    """For size independent growth dmu_10/dt = G1 mu_00 and dmu_01/dt = G2 mu_00.
    Both right hand sides are constant, so forward Euler integrates them exactly
    and the discrete moments match the closed form to roundoff."""
    grid = Grid(n1=64, n2=32, L1_max=640.0, L2_max=320.0)
    f0 = gaussian(grid, 150.0, 30.0, 90.0, 20.0, amp=100.0)
    u1, u2, t_end = 0.02, 0.01, 3000.0
    n_sub = substeps_for_cfl(grid, t_end, u1, u2, cfl=0.5)
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, t_end]),
                   ConstantRate(u1, u2), grid, n_substeps=n_sub, couple=False)
    mu00 = moment(f0, grid, 0, 0)
    for p, q, u in ((1, 0, u1), (0, 1, u2)):
        expected = moment(f0, grid, p, q) + u * mu00 * t_end
        got = moment(res.f_final, grid, p, q)
        assert float(abs(got - expected) / abs(expected)) < 1e-12


def test_second_moment_error_matches_the_predicted_truncation_term():
    """Donor cell upwinding gives dmu_20/dt = 2 G (mu_10 + dL1 mu_00 / 2), so the
    discrete mu_20 overshoots the continuum answer by exactly
    G mu_00 t dL1 (1 - Courant). Getting this term right pins the flux stencil
    and the face velocity locations, not just the leading order behaviour."""
    grid = Grid(n1=48, n2=24, L1_max=480.0, L2_max=240.0)
    f0 = gaussian(grid, 150.0, 30.0, 100.0, 25.0, amp=100.0)
    u1, t_end, n_sub = 0.02, 600.0, 20
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, t_end]),
                   ConstantRate(u1, 0.0), grid, n_substeps=n_sub, couple=False)
    dt = t_end / n_sub
    courant = u1 * dt / grid.dL1
    mu00, mu10, mu20 = (moment(f0, grid, *pq) for pq in ((0, 0), (1, 0), (2, 0)))
    continuum = mu20 + 2 * u1 * mu10 * t_end + u1**2 * mu00 * t_end**2
    predicted_excess = u1 * mu00 * t_end * grid.dL1 * (1.0 - courant)
    excess = moment(res.f_final, grid, 2, 0) - continuum
    assert float(excess / predicted_excess) == pytest.approx(1.0, rel=1e-9)


def test_no_influx_at_the_zero_length_boundary():
    """Eq (5): f = 0 at L = 0, and there is no nucleation term, so an empty grid
    under pure growth stays empty."""
    grid = Grid(n1=32, n2=16, L1_max=320.0, L2_max=160.0)
    f0 = jnp.zeros(grid.shape)
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, 3000.0]),
                   ConstantRate(0.05, 0.05), grid, n_substeps=60, couple=False)
    assert float(jnp.max(jnp.abs(res.f_final))) == 0.0


def test_no_influx_at_the_far_boundary_under_dissolution():
    """Eq (6): f = 0 at the far edge, so shrinking crystals cannot create
    population at large L. Starting from a seed that is exactly zero in the outer
    cells, dissolution has to leave them exactly zero."""
    grid = Grid(n1=32, n2=16, L1_max=320.0, L2_max=160.0)
    f0 = gaussian(grid, 100.0, 20.0, 50.0, 12.0, amp=100.0)
    f0 = f0.at[-6:, :].set(0.0).at[:, -4:].set(0.0)
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, 3000.0]),
                   ConstantRate(-0.02, -0.01), grid, n_substeps=60, couple=False)
    assert float(jnp.max(jnp.abs(res.f_final[-6:, :]))) == 0.0
    assert float(jnp.max(jnp.abs(res.f_final[:, -4:]))) == 0.0


def test_mass_leaves_the_grid_at_zero_length_under_dissolution():
    """The uncoupled solver loses mu_00 through L = 0 once crystals shrink to
    nothing, which is where the solute in eq (2) comes from."""
    grid = Grid(n1=32, n2=16, L1_max=320.0, L2_max=160.0)
    f0 = gaussian(grid, 40.0, 12.0, 20.0, 6.0, amp=100.0)
    res = simulate(f0, 0.0, T_FIXED, jnp.asarray([0.0, 4000.0]),
                   ConstantRate(-0.02, -0.01), grid, n_substeps=80, couple=False)
    assert float(moment(res.f_final, grid, 0, 0)) < 0.5 * float(moment(f0, grid, 0, 0))


def test_coupled_mass_balance_is_conserved():
    """Eq (2) integrated with the same flux divergence that updates f, so the
    total c + rho_c kv Vc is conserved to roundoff over a run that dissolves most
    of the seed."""
    grid = Grid(n1=64, n2=32, L1_max=1000.0, L2_max=200.0)
    f0 = lognormal_seed(grid, 350.0, 0.30, 60.0, 0.25, 1.0e12)
    t = jnp.linspace(0.0, 3600.0, 21)
    temperature = TemperatureProfile(times=t, values=jnp.linspace(28.0, 33.0, 21))
    c0 = concentration_from_S(0.98, 28.0)
    res = simulate(f0, c0, temperature, t, M1(), grid, n_substeps=8)
    before = total_mass(f0, c0, grid)
    after = total_mass(res.f_final, res.c[-1], grid)
    assert float(res.c[-1]) > float(c0)  # dissolution really happened
    assert float(crystal_volume(res.f_final, grid)) < 0.7 * 1.0e12
    assert float(abs(after - before) / before) < 1e-12


def test_concentration_gain_equals_crystal_mass_lost():
    """Same balance stated the other way round, so a wrong rho_c or kv shows up."""
    from hybrid_pbe.kinetics import K_V, RHO_C

    grid = Grid(n1=48, n2=24, L1_max=1000.0, L2_max=200.0)
    f0 = lognormal_seed(grid, 300.0, 0.30, 55.0, 0.25, 8.0e11)
    t = jnp.linspace(0.0, 3600.0, 11)
    c0 = concentration_from_S(0.98, 29.0)
    res = simulate(f0, c0, ConstantTemperature(29.0), t, M4(), grid, n_substeps=10)
    dv = crystal_volume(f0, grid) - crystal_volume(res.f_final, grid)
    assert float(res.c[-1] - c0) == pytest.approx(float(RHO_C * K_V * dv), rel=1e-10)


def test_f_stays_non_negative_under_the_cfl_condition():
    """Donor cell upwinding is positivity preserving for Courant <= 1, including
    across the sign change from dissolution to growth."""
    grid = Grid(n1=48, n2=24, L1_max=1200.0, L2_max=300.0)
    key = jax.random.PRNGKey(3)
    noisy = jnp.abs(jax.random.normal(key, grid.shape)) * gaussian(
        grid, 300.0, 80.0, 90.0, 30.0)
    f0 = noisy * (2.0e11 / crystal_volume(noisy, grid))
    t = jnp.linspace(0.0, 2400.0, 13)
    # Cooling ramp that runs from dissolution (S < 1) into growth (S > 1) while
    # staying inside the validity window [0.975, 1.25] of section 4.3.
    T_values = jnp.linspace(32.0, 27.0, 13)
    c0 = concentration_from_S(0.98, 32.0)
    res = simulate(f0, c0, TemperatureProfile(times=t, values=T_values), t,
                   GrowthDissolution(), grid, n_substeps=1)
    assert float(res.S.min()) < 1.0 < float(res.S.max()) < 1.25
    assert 0.5 < float(res.courant_max) < 1.0  # actually near the stability bound
    assert float(res.f_min) >= -1e-12 * float(jnp.max(f0))


def test_substeps_for_cfl_delivers_the_courant_number_it_promises():
    grid = Grid(n1=40, n2=20, L1_max=400.0, L2_max=200.0)
    t_end = 1800.0
    for cfl in (0.2, 0.5, 0.9):
        n_sub = substeps_for_cfl(grid, t_end, 0.03, 0.015, cfl=cfl)
        res = simulate(gaussian(grid, 200.0, 40.0, 100.0, 20.0), 0.0, T_FIXED,
                       jnp.asarray([0.0, t_end]), ConstantRate(0.03, 0.015), grid,
                       n_substeps=n_sub, couple=False)
        assert float(res.courant_max) <= cfl + 1e-12
        assert float(res.courant_max) > 0.5 * cfl


def test_dissolution_shrinks_the_population_and_raises_the_concentration():
    grid = Grid(n1=48, n2=24, L1_max=1000.0, L2_max=200.0)
    f0 = lognormal_seed(grid, 350.0, 0.30, 60.0, 0.25, 1.0e12)
    t = jnp.linspace(0.0, 3600.0, 11)
    res = simulate(f0, concentration_from_S(0.98, 30.0), ConstantTemperature(30.0),
                   t, M1(), grid, n_substeps=10)
    assert float(res.L1[-1]) < float(res.L1[0])
    assert float(res.L2[-1]) < float(res.L2[0])
    assert np.all(np.diff(np.asarray(res.c)) > 0)
    # M1 shrinks L1 about 2.32/0.19 times faster than L2, so needles get thinner
    # in relative terms but the aspect ratio falls.
    assert float(res.L1[0] - res.L1[-1]) > float(res.L2[0] - res.L2[-1])


def test_solution_is_independent_of_substep_count_to_first_order():
    """Halving dt at fixed grid changes the answer by well under a percent, so the
    trajectories reported elsewhere are grid limited rather than dt limited."""
    grid = Grid(n1=48, n2=24, L1_max=1000.0, L2_max=200.0)
    f0 = lognormal_seed(grid, 350.0, 0.30, 60.0, 0.25, 1.0e12)
    t = jnp.linspace(0.0, 3600.0, 11)
    c0 = concentration_from_S(0.98, 30.0)
    a = simulate(f0, c0, ConstantTemperature(30.0), t, M1(), grid, n_substeps=10)
    b = simulate(f0, c0, ConstantTemperature(30.0), t, M1(), grid, n_substeps=20)
    assert float(jnp.max(jnp.abs(a.L1 - b.L1)) / jnp.max(a.L1)) < 5e-3
    assert float(jnp.max(jnp.abs(a.c - b.c)) / jnp.max(a.c)) < 5e-3
