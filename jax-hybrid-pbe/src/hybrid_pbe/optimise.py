"""Process optimisation over T(t), section 4.6.

The model is frozen and the free variable phi becomes the temperature profile,
300 linearly spaced control values in the paper. Two tasks are here:

  optimise_trajectory   track a target (L1, L2) trajectory, eq (21)
  optimise_aspect_ratio reduce the product aspect ratio subject to a volume
                        floor, eq (22)

Constraints follow section 4.6: T is projected into [20, 41] C at every
iteration (hard), while |dT/dt| <= 0.05 C/min, S >= 0.975 and, where a growth law
is active, S <= 1.25 are soft penalties in eq (20). Penalty weights are ramped
over the run so the optimiser settles onto the boundaries instead of bouncing.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .grid import Grid, aspect_ratio, crystal_volume
from .kinetics import (
    DTDT_MAX,
    S_MAX,
    S_MIN,
    T_MAX_C,
    T_MIN_C,
    ramp_rate_penalty,
    supersaturation_penalty,
)
from .solver import TemperatureProfile, simulate


class OptResult(NamedTuple):
    T_values: jnp.ndarray
    history: jnp.ndarray
    objective: jnp.ndarray


def _run(T_values, t, f0, c0, kinetics, grid, n_substeps):
    profile = TemperatureProfile(times=t, values=T_values)
    return simulate(f0, c0, profile, t, kinetics, grid, n_substeps=n_substeps)


# P_S is a squared violation of a quantity of order one, P_Tdot a squared
# relative excess, so they need different scales before they are added.
RAMP_SCALE = 1e-4


def constraint_penalty(res, T_values, t, s_min=S_MIN, s_max=None, dtdt_max=DTDT_MAX):
    """P_S + P_Tdot of eq (20), before the ramped outer weight."""
    return (supersaturation_penalty(res.S, s_min, s_max)
            + RAMP_SCALE * ramp_rate_penalty(T_values, t, dtdt_max))


def optimise_aspect_ratio(T_init, t, f0, c0, kinetics, grid: Grid, steps: int = 200,
                          lr: float = 0.02, n_substeps: int = 6, s_min=S_MIN,
                          s_max=S_MAX, penalty_weight: float = 1e5,
                          volume_weight: float = 1e3) -> OptResult:
    """Minimise L1bar/L2bar at the end of the batch, eq (22).

    P_V holds the product volume at or above the seed volume, so the optimiser
    cannot cheat by dissolving everything. With the growth law's L1^1.2 size
    dependence and no size dependence in G2, shrinking the population first and
    growing it back is what lowers the ratio, which is the mechanism the paper
    reports in section 2.3.2.
    """
    v_seed = crystal_volume(f0, grid)

    def objective(T_values, weight):
        res = _run(T_values, t, f0, c0, kinetics, grid, n_substeps)
        ar = aspect_ratio(res.f_final, grid)
        v_ratio = crystal_volume(res.f_final, grid) / v_seed
        p_v = volume_weight * jax.nn.relu(1.0 - v_ratio) ** 2
        return ar + p_v + weight * constraint_penalty(res, T_values, t, s_min, s_max)

    return _run_adam(objective, T_init, steps, lr, penalty_weight,
                     lambda T: objective(T, penalty_weight))


def optimise_trajectory(T_init, t, f0, c0, kinetics, grid: Grid, L1_target, L2_target,
                        steps: int = 200, lr: float = 0.02, n_substeps: int = 6,
                        s_min=S_MIN, s_max=None, penalty_weight: float = 1e5) -> OptResult:
    """Recover T(t) from a target size trajectory, eq (21)."""

    def objective(T_values, weight):
        res = _run(T_values, t, f0, c0, kinetics, grid, n_substeps)
        mse = 0.5 * (jnp.mean((res.L1 - L1_target) ** 2) + jnp.mean((res.L2 - L2_target) ** 2))
        return mse + weight * constraint_penalty(res, T_values, t, s_min, s_max)

    return _run_adam(objective, T_init, steps, lr, penalty_weight,
                     lambda T: objective(T, penalty_weight))


def _run_adam(objective, T_init, steps, lr, penalty_weight, final_objective):
    """Adam on T(t) with a hard box projection into [20, 41] C after every update."""
    T = jnp.asarray(T_init, dtype=float)
    opt = optax.adam(lr)
    state = opt.init(T)

    @eqx.filter_jit
    def step(T, state, weight):
        loss, g = jax.value_and_grad(objective)(T, weight)
        updates, state = opt.update(g, state, T)
        T = jnp.clip(T + updates, T_MIN_C, T_MAX_C)
        return T, state, loss

    history = []
    for i in range(steps):
        # A jnp scalar, not a python float: eqx.filter_jit treats python scalars
        # as static and would recompile on every iteration of the ramp.
        weight = jnp.asarray(penalty_weight * (0.1 + 0.9 * (i + 1) / steps))
        T, state, loss = step(T, state, weight)
        history.append(loss)
    return OptResult(T_values=T, history=jnp.asarray(history),
                     objective=final_objective(T))
