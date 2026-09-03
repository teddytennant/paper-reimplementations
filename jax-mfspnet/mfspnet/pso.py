"""Particle swarm optimization over the fixed-length encoding (Sec. 2.2, Sec. 4.5).

Velocity is Eq. 4 and position is Eq. 5:

    v_i^d(t+1) = w v_i^d(t) + c1 rho1 (p_i^d - x_i^d) + c2 rho2 (g^d - x_i^d)
    x_i^d(t+1) = x_i^d(t) + v_i^d(t+1)

with rho1, rho2 drawn per dimension from U[0, 1]. Table 2 gives
w = 0.83, c1 = 1.17, c2 = 1.69, and those are the defaults here. Appendix C's
Table 9 disagrees with itself about this: its last row marks "balanced
c1 = c2 = 1.49" as the value used. Table 2 is the one labelled as the
experiment settings, so Table 2 wins.

There is no special handling of block-length mismatch between particles, and
that is deliberate: Sec. 4.5 says the existing PSO procedure can be used
because the flag encoding turns variable block length into a fixed-length
vector. A coordinate crossing r_lower in either direction is how a layer gets
switched on or off.

Velocity clamping and boundary handling are not given in the paper. Positions
are clamped to [S_v, r_upper] because anything outside that is not decodable,
and a coordinate that hits a wall has its velocity zeroed (an absorbing wall).
Without the absorbing wall a swarm parks on the bound with the velocity still
pointing outward and the run stalls there. Velocity clamping is available but
off unless a limit is passed.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from .encoding import SearchSpace, initialize_population

INERTIA = 0.83
COGNITIVE = 1.17
SOCIAL = 1.69


@dataclasses.dataclass
class PSOConfig:
    """PSO control parameters. Defaults are Table 2."""

    inertia: float = INERTIA
    cognitive: float = COGNITIVE
    social: float = SOCIAL
    v_max: float | None = None
    clamp_position: bool = True


@dataclasses.dataclass
class SwarmState:
    position: np.ndarray
    velocity: np.ndarray
    pbest: np.ndarray
    pbest_fitness: np.ndarray
    gbest: np.ndarray
    gbest_fitness: float


def velocity_update(
    velocity: np.ndarray,
    position: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray,
    cfg: PSOConfig,
) -> np.ndarray:
    """Eq. 4, with the two random draws supplied so the step is testable."""
    inertia = cfg.inertia * velocity
    cognitive = cfg.cognitive * rho1 * (pbest - position)
    social = cfg.social * rho2 * (gbest[None, :] - position)
    v = inertia + cognitive + social
    if cfg.v_max is not None:
        v = np.clip(v, -cfg.v_max, cfg.v_max)
    return v


def position_update(
    position: np.ndarray, velocity: np.ndarray, space: SearchSpace, cfg: PSOConfig
):
    """Eq. 5, then clamp into the decodable interval.

    Returns the new position and the velocity after the absorbing wall, so a
    coordinate that ran into a bound restarts from zero rather than staying
    pinned by its own momentum.
    """
    x = position + velocity
    if not cfg.clamp_position:
        return x, velocity
    clipped = space.clip(x)
    velocity = np.where(clipped != x, 0.0, velocity)
    return clipped, velocity


def step(
    state: SwarmState,
    space: SearchSpace,
    cfg: PSOConfig,
    rng: np.random.Generator,
) -> SwarmState:
    """One velocity and position update for the whole swarm. No fitness here."""
    shape = state.position.shape
    rho1 = rng.random(shape)
    rho2 = rng.random(shape)
    v = velocity_update(
        state.velocity, state.position, state.pbest, state.gbest, rho1, rho2, cfg
    )
    x, v = position_update(state.position, v, space, cfg)
    return dataclasses.replace(state, position=x, velocity=v)


def update_bests(state: SwarmState, fitness: np.ndarray) -> SwarmState:
    """Eqs. 2 and 3 for a minimization objective (Alg. 3, lines 11-16).

    A particle keeps its personal best unless the new position strictly beats
    it, which is the rule stated in Sec. 4.5, point 1.
    """
    improved = fitness < state.pbest_fitness
    pbest = np.where(improved[:, None], state.position, state.pbest)
    pbest_fitness = np.where(improved, fitness, state.pbest_fitness)
    best = int(np.argmin(pbest_fitness))
    return dataclasses.replace(
        state,
        pbest=pbest,
        pbest_fitness=pbest_fitness,
        gbest=pbest[best].copy(),
        gbest_fitness=float(pbest_fitness[best]),
    )


def initial_state(
    positions: np.ndarray, fitness: np.ndarray, velocity: np.ndarray | None = None
) -> SwarmState:
    """Alg. 3, lines 3-4: pbest starts at the initial position."""
    positions = np.asarray(positions, dtype=np.float64)
    fitness = np.asarray(fitness, dtype=np.float64)
    if velocity is None:
        velocity = np.zeros_like(positions)
    best = int(np.argmin(fitness))
    return SwarmState(
        position=positions.copy(),
        velocity=np.asarray(velocity, dtype=np.float64).copy(),
        pbest=positions.copy(),
        pbest_fitness=fitness.copy(),
        gbest=positions[best].copy(),
        gbest_fitness=float(fitness[best]),
    )


def minimize(
    objective: Callable[[np.ndarray], np.ndarray],
    space: SearchSpace,
    n_particles: int,
    n_generations: int,
    rng: np.random.Generator,
    cfg: PSOConfig | None = None,
    positions: np.ndarray | None = None,
):
    """Run the swarm on a callable objective and return (gbest, history).

    ``objective`` takes the (n_particles, l_max) position array and returns one
    fitness per particle, lower is better. Alg. 3 wires this to VLE-EMA; the
    PSO mechanics do not care what the fitness is.
    """
    cfg = cfg or PSOConfig()
    if positions is None:
        positions = initialize_population(rng, n_particles, space)
    positions = space.clip(np.asarray(positions, dtype=np.float64))
    fitness = np.asarray(objective(positions), dtype=np.float64)
    state = initial_state(positions, fitness)
    history = [state.gbest_fitness]
    for _ in range(n_generations):
        state = step(state, space, cfg, rng)
        fitness = np.asarray(objective(state.position), dtype=np.float64)
        state = update_bests(state, fitness)
        history.append(state.gbest_fitness)
    return state, np.asarray(history)
