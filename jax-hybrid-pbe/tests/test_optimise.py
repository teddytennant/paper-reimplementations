"""Process optimisation over T(t), section 4.6.

Nothing here is stochastic: the solver, the objective and Adam are all
deterministic, so the only free choice is the starting profile. Each check runs
from three different initialisations to make sure the result is not an artefact
of one of them.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from hybrid_pbe.data import lognormal_seed
from hybrid_pbe.grid import Grid, aspect_ratio, crystal_volume
from hybrid_pbe.kinetics import (
    DTDT_MAX,
    M1,
    S_MAX,
    S_MIN,
    T_MAX_C,
    T_MIN_C,
    GrowthDissolution,
    concentration_from_S,
)
from hybrid_pbe.optimise import constraint_penalty, optimise_aspect_ratio, optimise_trajectory
from hybrid_pbe.solver import TemperatureProfile, simulate

GRID = Grid(n1=32, n2=16, L1_max=1200.0, L2_max=250.0)


def _run(T_values, t, f0, c0, kinetics, n_substeps):
    return simulate(f0, c0, TemperatureProfile(times=t, values=T_values), t,
                    kinetics, GRID, n_substeps=n_substeps)


# --- trajectory optimisation, section 4.6.1 ---------------------------------

T_TRAJ = jnp.linspace(0.0, 7200.0, 13)
T_TRUE = 30.0 + 0.8 * jnp.tanh((T_TRAJ - 3600.0) / 1200.0)
F0_TRAJ = lognormal_seed(GRID, 400.0, 0.30, 62.0, 0.25, 8.0e11)
C0_TRAJ = concentration_from_S(0.99, float(T_TRUE[0]))


@pytest.fixture(scope="module")
def target():
    res = _run(T_TRUE, T_TRAJ, F0_TRAJ, C0_TRAJ, M1(), 6)
    # The synthetic target has to stay inside M1's fitted range or the
    # constraint penalty would be fighting the tracking term.
    assert float(res.S.min()) > S_MIN
    assert float(res.S.max()) < 1.0
    return res


def test_trajectory_optimisation_recovers_the_temperature_profile(target):
    """Section 4.6.1 on synthetic data: given only the (L1, L2) trajectories,
    gradient descent on 13 control temperatures gets back the profile that made
    them, from initial guesses offset by a whole degree either way."""
    for shift in (-1.0, 0.0, 1.0):
        guess = jnp.linspace(float(T_TRUE[0]) + shift, float(T_TRUE[-1]) + shift, 13)
        out = optimise_trajectory(guess, T_TRAJ, F0_TRAJ, C0_TRAJ, M1(), GRID,
                                  target.L1, target.L2, steps=250, lr=0.05,
                                  n_substeps=6)
        before = float(jnp.mean(jnp.abs(guess - T_TRUE)))
        after = float(jnp.mean(jnp.abs(out.T_values - T_TRUE)))
        assert after < 0.05, (shift, after)
        assert after < 0.2 * before, (shift, before, after)
        assert float(out.history[-1]) < 1e-2 * float(out.history[0])


def test_optimised_profile_respects_the_hard_temperature_box(target):
    """T is projected into [20, 41] C after every Adam update, section 4.6."""
    guess = jnp.full((13,), 19.0)  # starts below the box
    out = optimise_trajectory(guess, T_TRAJ, F0_TRAJ, C0_TRAJ, M1(), GRID,
                              target.L1, target.L2, steps=40, lr=0.5, n_substeps=6)
    assert float(out.T_values.min()) >= T_MIN_C
    assert float(out.T_values.max()) <= T_MAX_C


# --- product aspect ratio optimisation, section 4.6.2 -----------------------

T_AR = jnp.linspace(0.0, 43200.0, 19)
F0_AR = lognormal_seed(GRID, 400.0, 0.30, 62.0, 0.25, 3.0e11)
C0_AR = concentration_from_S(0.98, 33.0)
V_SEED = float(crystal_volume(F0_AR, GRID))
AR_SEED = float(aspect_ratio(F0_AR, GRID))


@pytest.fixture(scope="module")
def optimised_profiles():
    out = {}
    for T_end in (25.0, 27.0, 29.0):
        guess = jnp.linspace(33.0, T_end, 19)
        res = optimise_aspect_ratio(guess, T_AR, F0_AR, C0_AR, GrowthDissolution(),
                                    GRID, steps=200, lr=0.05, n_substeps=8)
        out[T_end] = (guess, res)
    return out


def test_aspect_ratio_optimisation_reduces_the_product_aspect_ratio(optimised_profiles):
    """Eq (22). The paper takes 6.50 to 4.88 on the real seed of E5; on this
    synthetic seed of aspect ratio 6.31 the three runs land between 4.70 and
    5.14, a 19 to 26 percent reduction."""
    for T_end, (_, res) in optimised_profiles.items():
        final = _run(res.T_values, T_AR, F0_AR, C0_AR, GrowthDissolution(), 8)
        ratio = float(aspect_ratio(final.f_final, GRID))
        assert ratio < 0.85 * AR_SEED, (T_end, ratio, AR_SEED)
        assert float(res.history[-1]) < float(res.history[0])


def test_optimised_profile_keeps_the_product_volume(optimised_profiles):
    """P_V of eq (22): Vproduct / Vseed >= 1, a soft penalty, so a fraction of a
    percent under the floor is allowed."""
    for T_end, (_, res) in optimised_profiles.items():
        final = _run(res.T_values, T_AR, F0_AR, C0_AR, GrowthDissolution(), 8)
        v_ratio = float(crystal_volume(final.f_final, GRID)) / V_SEED
        assert v_ratio > 0.98, (T_end, v_ratio)


def test_optimised_profile_stays_within_the_kinetic_validity_range(optimised_profiles):
    """Smin = 0.975 and Smax = 1.25 of section 4.3, both soft, so a small
    overshoot at the floor is expected; the ceiling is never approached."""
    for T_end, (_, res) in optimised_profiles.items():
        final = _run(res.T_values, T_AR, F0_AR, C0_AR, GrowthDissolution(), 8)
        assert float(final.S.min()) > S_MIN - 5e-3, (T_end, float(final.S.min()))
        assert float(final.S.max()) < S_MAX, (T_end, float(final.S.max()))
        assert float(res.T_values.min()) >= T_MIN_C
        assert float(res.T_values.max()) <= T_MAX_C


def test_optimised_profile_dissolves_before_it_grows(optimised_profiles):
    """The mechanism of section 2.3.2: shrink the population first so the L1**1.2
    size dependence of G1 has less to work with, then grow the mass back mostly
    into the width. Every run here starts under S = 1 and ends over it."""
    for T_end, (_, res) in optimised_profiles.items():
        final = _run(res.T_values, T_AR, F0_AR, C0_AR, GrowthDissolution(), 8)
        assert float(final.S[0]) < 1.0, T_end
        assert float(final.S[-1]) > 1.0, T_end
        # The mass ends up in the width: L2 grows relative to L1.
        assert float(final.L2[-1]) / float(final.L2[0]) > \
            float(final.L1[-1]) / float(final.L1[0]), T_end


def test_optimised_profile_respects_the_ramp_rate_limit(optimised_profiles):
    """|dT/dt| <= 0.05 C/min is a soft penalty, so this checks the profile is at
    the limit rather than far past it."""
    for T_end, (_, res) in optimised_profiles.items():
        rate = jnp.abs(jnp.diff(res.T_values) / jnp.diff(T_AR))
        assert float(jnp.max(rate)) < 1.2 * DTDT_MAX, (T_end, float(jnp.max(rate)))


def test_constraint_penalty_vanishes_on_a_feasible_profile():
    t = jnp.linspace(0.0, 43200.0, 19)
    T_values = jnp.linspace(30.0, 29.0, 19)  # well inside the ramp limit
    res = _run(T_values, t, F0_AR, concentration_from_S(0.99, 30.0),
               GrowthDissolution(), 8)
    assert float(res.S.min()) >= S_MIN and float(res.S.max()) <= S_MAX
    assert float(constraint_penalty(res, T_values, t, S_MIN, S_MAX)) == 0.0

    # P_S bites when the batch drops under the dissolution floor.
    starved = _run(T_values, t, F0_AR, concentration_from_S(0.94, 30.0),
                   GrowthDissolution(), 8)
    assert float(starved.S.min()) < S_MIN
    assert float(constraint_penalty(starved, T_values, t, S_MIN, S_MAX)) > 0.0

    # P_Tdot bites on a ramp the jacket could not deliver: 9 C in 40 minutes.
    t_short = jnp.linspace(0.0, 2400.0, 19)
    fast = jnp.linspace(33.0, 24.0, 19)
    res_fast = _run(fast, t_short, F0_AR, concentration_from_S(0.99, 33.0),
                    GrowthDissolution(), 8)
    assert float(jnp.max(jnp.abs(jnp.diff(fast) / jnp.diff(t_short)))) > DTDT_MAX
    assert float(constraint_penalty(res_fast, fast, t_short, S_MIN, S_MAX)) > 0.0
