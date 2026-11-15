"""Differentiable finite volume solver for the bivariate PBE, eqs (1) to (6).

Space: first order upwind fluxes on the uniform grid of grid.Grid. The paper uses
a high resolution FVM; this is the plain donor cell scheme, so it is more
diffusive but has the same conservation structure and stays positive under the
CFL condition. See the README.

Time: forward Euler. Each observation interval is split into a fixed number of
substeps so the whole trajectory is a lax.scan of static length and reverse mode
differentiation works without a while loop. Pick the substep count with
substeps_for_cfl; every run reports the largest Courant number it actually hit.

Boundaries: the ghost cells outside [0, L1_max] x [0, L2_max] hold f = 0, which
is exactly eqs (5) and (6). Under growth nothing enters at L = 0 (no nucleation
term here); under dissolution crystals leave through L = 0 and their mass shows
up in the liquid through eq (2).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .grid import Grid, crystal_volume, mean_lengths, moment
from .kinetics import K_V, RHO_C, supersaturation


class SimResult(NamedTuple):
    """Trajectory of the three observed predictors plus the final state."""

    t: jnp.ndarray  # (n_obs,)
    c: jnp.ndarray  # (n_obs,) solute concentration [g/kg]
    L1: jnp.ndarray  # (n_obs,) volume weighted mean length [um]
    L2: jnp.ndarray  # (n_obs,) volume weighted mean width [um]
    S: jnp.ndarray  # (n_obs,) supersaturation [-]
    f_final: jnp.ndarray  # (n1, n2)
    courant_max: jnp.ndarray  # scalar, largest Courant number over the run
    f_min: jnp.ndarray  # scalar, smallest f seen at any step, for positivity checks


def face_velocities(kinetics, grid: Grid, S, T):
    """U1 on the (n1+1, n2) L1-faces and U2 on the (n1, n2+1) L2-faces."""
    n1, n2 = grid.shape
    l1e = jnp.broadcast_to(grid.L1_edges[:, None], (n1 + 1, n2))
    l2c = jnp.broadcast_to(grid.L2_centers[None, :], (n1 + 1, n2))
    u1, _ = kinetics(l1e, l2c, S, T)

    l1c = jnp.broadcast_to(grid.L1_centers[:, None], (n1, n2 + 1))
    l2e = jnp.broadcast_to(grid.L2_edges[None, :], (n1, n2 + 1))
    _, u2 = kinetics(l1c, l2e, S, T)
    return u1, u2


def flux_divergence(f, u1, u2, grid: Grid):
    """-div(U f) with donor cell upwinding and f = 0 in the ghost cells.

    Returns df/dt for the homogeneous PBE, eq (1).
    """
    n1, n2 = grid.shape
    zeros1 = jnp.zeros((1, n2), dtype=f.dtype)
    left1 = jnp.concatenate([zeros1, f], axis=0)  # cell on the low side of each face
    right1 = jnp.concatenate([f, zeros1], axis=0)  # cell on the high side
    F1 = u1 * jnp.where(u1 > 0, left1, right1)

    zeros2 = jnp.zeros((n1, 1), dtype=f.dtype)
    left2 = jnp.concatenate([zeros2, f], axis=1)
    right2 = jnp.concatenate([f, zeros2], axis=1)
    F2 = u2 * jnp.where(u2 > 0, left2, right2)

    return -((F1[1:, :] - F1[:-1, :]) / grid.dL1 + (F2[:, 1:] - F2[:, :-1]) / grid.dL2)


def euler_step(f, c, T, dt, kinetics, grid: Grid, couple: bool = True):
    """One forward Euler step of the coupled system, eqs (1) and (2).

    The liquid phase update uses dVc/dt taken straight from the discrete flux
    divergence, so c + rho_c kv Vc is conserved to roundoff rather than to
    truncation error.

    One caveat: any crystal mass that leaves through the far edge is credited
    back to the liquid, which is only right because the domain is sized so f is
    zero there. test_no_influx_at_the_far_boundary_under_dissolution pins the
    boundary condition itself; sizing the grid is on the caller.
    """
    S = supersaturation(c, T)
    u1, u2 = face_velocities(kinetics, grid, S, T)
    dfdt = flux_divergence(f, u1, u2, grid)
    f_new = f + dt * dfdt
    if couple:
        dVcdt = moment(dfdt, grid, 1, 2)
        c_new = c - dt * RHO_C * K_V * dVcdt
    else:
        c_new = c
    courant = dt * (jnp.max(jnp.abs(u1)) / grid.dL1 + jnp.max(jnp.abs(u2)) / grid.dL2)
    return f_new, c_new, courant


def observe(f, c, T, grid: Grid):
    l1, l2 = mean_lengths(f, grid)
    return c, l1, l2, supersaturation(c, T)


@eqx.filter_jit
def simulate(f0, c0, temperature, t_eval, kinetics, grid: Grid, n_substeps: int = 8,
             couple: bool = True) -> SimResult:
    """Integrate eqs (1) to (6) and report the predictors at t_eval.

    temperature is any callable t -> T [C]; TemperatureProfile below is the
    differentiable piecewise linear one used for process optimisation.
    t_eval must be increasing with t_eval[0] the initial time.
    """
    t_eval = jnp.asarray(t_eval, dtype=f0.dtype)
    dts = jnp.diff(t_eval) / n_substeps

    def interval(carry, dt):
        f, c, t, cmax, fmin = carry
        obs = observe(f, c, temperature(t), grid)

        def sub(inner, _):
            f, c, t, cmax, fmin = inner
            T = temperature(t)
            f, c, courant = euler_step(f, c, T, dt, kinetics, grid, couple)
            return (f, c, t + dt, jnp.maximum(cmax, courant), jnp.minimum(fmin, jnp.min(f))), None

        carry, _ = jax.lax.scan(sub, (f, c, t, cmax, fmin), None, length=n_substeps)
        return carry, obs

    zero = jnp.zeros((), dtype=f0.dtype)
    init = (f0, jnp.asarray(c0, dtype=f0.dtype), t_eval[0], zero, jnp.min(f0))
    (f, c, t, cmax, fmin), obs = jax.lax.scan(interval, init, dts)
    last = observe(f, c, temperature(t), grid)
    stacked = [jnp.concatenate([o, jnp.reshape(l, (1,))]) for o, l in zip(obs, last)]
    return SimResult(t=t_eval, c=stacked[0], L1=stacked[1], L2=stacked[2], S=stacked[3],
                     f_final=f, courant_max=cmax, f_min=fmin)


def substeps_for_cfl(grid: Grid, dt_interval, u1_max, u2_max, cfl: float = 0.5) -> int:
    """Smallest substep count per interval that keeps the Courant number below cfl.

    u1_max and u2_max are bounds on |U1| and |U2| over the run. The 2D upwind
    stability condition is dt(|U1|/dL1 + |U2|/dL2) <= 1.
    """
    rate = float(u1_max) / grid.dL1 + float(u2_max) / grid.dL2
    if rate <= 0.0:
        return 1
    return max(1, int(jnp.ceil(float(dt_interval) * rate / cfl)))


class TemperatureProfile(eqx.Module):
    """Piecewise linear T(t) through a set of control values.

    Section 4.6 parametrises the operating condition by 300 linearly spaced
    temperatures. values is the free variable phi in that task.
    """

    times: jnp.ndarray
    values: jnp.ndarray

    def __call__(self, t):
        return jnp.interp(t, self.times, self.values)


class ConstantTemperature(eqx.Module):
    value: jnp.ndarray

    def __init__(self, value):
        self.value = jnp.asarray(value, dtype=float)

    def __call__(self, t):
        return jnp.broadcast_to(self.value, jnp.shape(t))


def total_mass(f, c, grid: Grid):
    """Solute mass plus crystal mass per kg of solvent [g/kg].

    Eq (2) says this is constant. Crystals that dissolve past L = 0 leave the
    grid, and their mass has already been handed to the liquid phase by then.
    """
    return c + RHO_C * K_V * crystal_volume(f, grid)
