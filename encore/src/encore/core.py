"""ENCORE on a 3-state discrete chain.

Levels n = 0..N. Level N is the reference. A state at level n is a trajectory
stored reference-first, (x_N, ..., x_n), whose endpoint is x_n.

The upward proposal truncates the lower trajectory. The downward proposal
extends the upper trajectory by one draw from Q. The swap is accepted with
the Metropolis-Hastings probability in Eq. 3.5. When Q = P the density ratio
is 1. If every potential is constant and Q = P, every exchange is accepted.

This module does not implement a diffusion SDE.
"""

from __future__ import annotations

import itertools
import os
from typing import NamedTuple, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

Array = jax.Array

N_STATES = 3
N_LEVELS = 2


class EncoreRun(NamedTuple):
    """Level-0 endpoints and the exchange decisions from one ENCORE run."""

    endpoints: Array
    alphas: Array
    accepted: Array
    n_rejections: int
    n_proposals: int


def reference_eta(n_states: int = N_STATES) -> Array:
    """Uniform reference distribution on the state space."""
    return jnp.ones(n_states, dtype=jnp.float64) / n_states


def backward_kernel(n_states: int = N_STATES) -> Array:
    """Fixed row-stochastic backward kernel, shared by every level.

    Entry [i, j] is P(x_{n-1} = j | x_n = i).
    """
    if n_states != N_STATES:
        raise ValueError("the synthetic model uses 3 states")
    kernel = np.array(
        [
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.125, 0.375, 0.5],
        ],
        dtype=np.float64,
    )
    return jnp.asarray(kernel, dtype=jnp.float64)


def tilted_potentials() -> tuple[Array, Array, Array]:
    """G_0, G_1, G_2. G_2 is 1. G_0 and G_1 are positive and non-constant."""
    raw = (
        [1.0, 2.0, 0.5],
        [0.5, 1.5, 1.0],
        [1.0, 1.0, 1.0],
    )
    return tuple(jnp.asarray(g, dtype=jnp.float64) for g in raw)


def constant_potentials(
    n_levels: int = N_LEVELS,
    n_states: int = N_STATES,
    value: float = 1.0,
) -> tuple[Array, ...]:
    """Constant potentials, one array per level from 0 through n_levels."""
    g = jnp.full(n_states, value, dtype=jnp.float64)
    return tuple(g for _ in range(n_levels + 1))


def enumerate_trajectories(length: int, n_states: int = N_STATES) -> list[tuple[int, ...]]:
    """Every reference-first trajectory of the given length, in lexicographic order."""
    if length < 1:
        raise ValueError("trajectory length must be positive")
    return list(itertools.product(range(n_states), repeat=length))


def truncate(trajectory: Array) -> Array:
    """Drop the endpoint. This is the upward proposal."""
    trajectory = jnp.asarray(trajectory)
    if trajectory.shape[0] < 2:
        raise ValueError("cannot truncate a length-1 trajectory")
    return trajectory[:-1]


def extend(trajectory: Array, state: int) -> Array:
    """Append one state. This is the downward proposal."""
    trajectory = jnp.asarray(trajectory, dtype=jnp.int32)
    extra = jnp.asarray([state], dtype=jnp.int32)
    return jnp.concatenate([trajectory, extra])


def path_density(trajectory: Sequence[int], eta: Array, kernel: Array) -> float:
    """Generative path density eta(x_N) times the backward kernel along the path."""
    eta_np = np.asarray(eta, dtype=np.float64)
    kernel_np = np.asarray(kernel, dtype=np.float64)
    density = float(eta_np[trajectory[0]])
    for parent, child in zip(trajectory[:-1], trajectory[1:]):
        density *= float(kernel_np[parent, child])
    return density


def pushforward_marginal(eta: Array, kernel: Array, n_steps: int) -> Array:
    """Push eta through n_steps of a row-stochastic kernel. p_0 uses n_steps = N."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    marginal = jnp.asarray(eta, dtype=jnp.float64)
    kernel = jnp.asarray(kernel, dtype=jnp.float64)
    for _ in range(n_steps):
        marginal = marginal @ kernel
    return marginal


def endpoint_target(potential: Array, marginal: Array) -> Array:
    """Normalize G(x) * p(x) into the level-wise endpoint target."""
    unnorm = jnp.asarray(potential, dtype=jnp.float64) * jnp.asarray(marginal, dtype=jnp.float64)
    total = jnp.sum(unnorm)
    if not np.isfinite(float(total)) or float(total) <= 0.0:
        raise ValueError("endpoint target has non-positive mass")
    return unnorm / total


def exchange_weight(
    g_lower_end: float,
    g_upper_end: float,
    p_cond: float = 1.0,
    q_cond: float = 1.0,
) -> float:
    """Weight w(lower, upper) = (G_{n-1}(x_{n-1}) / G_n(x_n)) * (P / Q).

    Returns 0 when the numerator is 0 and the denominator is positive.
    Returns inf when the denominator is 0 and the numerator is positive.
    Returns 0 when both are 0; the acceptance ratio treats 0/0 as 1.
    """
    numerator = float(g_lower_end) * float(p_cond)
    denominator = float(g_upper_end) * float(q_cond)
    if numerator == 0.0 and denominator == 0.0:
        return 0.0
    if denominator == 0.0:
        return float(np.inf)
    return numerator / denominator


def acceptance_probability(
    g_lower: Array,
    g_upper: Array,
    x_lower_end: int,
    x_upper_end: int,
    y_lower_end: int,
    y_upper_end: int,
    kernel_p: Array | None = None,
    kernel_q: Array | None = None,
) -> Array:
    """Metropolis-Hastings probability min(1, w(y_lower, x_upper) / w(x_lower, y_upper)).

    Omit both kernels when Q = P, so the density ratio is 1. If both weights
    are 0, the ratio is taken to be 1. A positive-over-zero ratio is capped at 1.
    """
    if (kernel_p is None) ^ (kernel_q is None):
        raise ValueError("pass both kernels or neither")
    g_lower_np = np.asarray(g_lower, dtype=np.float64)
    g_upper_np = np.asarray(g_upper, dtype=np.float64)
    x_l = int(x_lower_end)
    x_u = int(x_upper_end)
    y_l = int(y_lower_end)
    y_u = int(y_upper_end)
    if kernel_p is None:
        p_fwd = q_fwd = p_rev = q_rev = 1.0
    else:
        kernel_p_np = np.asarray(kernel_p, dtype=np.float64)
        kernel_q_np = np.asarray(kernel_q, dtype=np.float64)
        p_fwd = float(kernel_p_np[x_u, y_l])
        q_fwd = float(kernel_q_np[x_u, y_l])
        p_rev = float(kernel_p_np[y_u, x_l])
        q_rev = float(kernel_q_np[y_u, x_l])

    # w(y_lower, x_upper) / w(x_lower, y_upper), written as a single fraction
    # so a 0/0 weight ratio is detected before any division.
    num = (
        float(g_lower_np[y_l])
        * float(g_upper_np[y_u])
        * p_fwd
        * q_rev
    )
    den = (
        float(g_upper_np[x_u])
        * float(g_lower_np[x_l])
        * q_fwd
        * p_rev
    )
    if num == 0.0 and den == 0.0:
        alpha = 1.0
    elif den == 0.0:
        alpha = 1.0
    elif num == 0.0:
        alpha = 0.0
    else:
        alpha = min(1.0, num / den)
    return jnp.asarray(alpha, dtype=jnp.float64)


def init_trajectories(
    key: Array,
    potentials: Sequence[Array],
    kernel: Array,
    eta: Array | None = None,
) -> list[Array]:
    """Independent generative trajectories, one per level, reference-first."""
    kernel = jnp.asarray(kernel, dtype=jnp.float64)
    n_states = int(kernel.shape[0])
    if eta is None:
        eta = reference_eta(n_states)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    n_max = len(potentials) - 1
    keys = jax.random.split(key, n_max + 1)
    trajectories = []
    for level in range(n_max + 1):
        length = n_max - level + 1
        trajectories.append(_sample_path(keys[level], length, kernel, eta))
    return trajectories


def run_encore(
    key: Array,
    n_iterations: int,
    potentials: Sequence[Array],
    kernel_p: Array,
    kernel_q: Array | None = None,
    eta: Array | None = None,
    initial: Sequence[Array] | None = None,
) -> EncoreRun:
    """Algorithm 1. Returns level-0 endpoints and every exchange decision.

    Each iteration draws a fresh reference state, then exchanges every adjacent
    pair whose upper index is congruent to the iteration index modulo 2.
    Iterations are numbered from 1.
    """
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")
    potentials = [jnp.asarray(g, dtype=jnp.float64) for g in potentials]
    kernel_p = jnp.asarray(kernel_p, dtype=jnp.float64)
    if kernel_q is None:
        kernel_q = kernel_p
    else:
        kernel_q = jnp.asarray(kernel_q, dtype=jnp.float64)
    n_states = int(kernel_p.shape[0])
    if eta is None:
        eta = reference_eta(n_states)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    n_max = len(potentials) - 1
    if initial is None:
        key, sub = jax.random.split(key)
        initial = init_trajectories(sub, potentials, kernel_p, eta)
    trajectories = [jnp.asarray(traj, dtype=jnp.int32) for traj in initial]
    _check_trajectory_lengths(trajectories, n_max)

    endpoints: list[int] = []
    alphas: list[float] = []
    accepted_flags: list[bool] = []
    for m in range(1, n_iterations + 1):
        key, sub = jax.random.split(key)
        x_ref = int(jax.random.choice(sub, n_states, p=eta))
        trajectories[n_max] = jnp.asarray([x_ref], dtype=jnp.int32)
        for n in range(1, n_max + 1):
            if n % 2 != m % 2:
                continue
            key, sub = jax.random.split(key)
            lower, upper, alpha, accepted = _exchange(
                trajectories[n - 1],
                trajectories[n],
                sub,
                potentials[n - 1],
                potentials[n],
                kernel_p,
                kernel_q,
            )
            trajectories[n - 1] = lower
            trajectories[n] = upper
            alphas.append(alpha)
            accepted_flags.append(accepted)
        endpoints.append(int(trajectories[0][-1]))

    if alphas:
        alpha_arr = jnp.asarray(alphas, dtype=jnp.float64)
        accepted_arr = jnp.asarray(accepted_flags, dtype=jnp.bool_)
        n_rejections = int(np.count_nonzero(~np.asarray(accepted_flags)))
    else:
        alpha_arr = jnp.zeros((0,), dtype=jnp.float64)
        accepted_arr = jnp.zeros((0,), dtype=jnp.bool_)
        n_rejections = 0
    return EncoreRun(
        endpoints=jnp.asarray(endpoints, dtype=jnp.int32),
        alphas=alpha_arr,
        accepted=accepted_arr,
        n_rejections=n_rejections,
        n_proposals=len(alphas),
    )


def exchange_kernel(
    level_upper: int,
    potentials: Sequence[Array],
    kernel_p: Array,
    kernel_q: Array | None = None,
) -> tuple[np.ndarray, list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Metropolis kernel of one adjacent exchange, on enumerated trajectory pairs.

    Rows are indexed by i_lower * n_upper + i_upper, in the same order as
    enumerate_trajectories. The kernel does not depend on eta. The matching
    stationary measure is the product of the two trajectory targets.
    """
    potentials_np = [np.asarray(g, dtype=np.float64) for g in potentials]
    kernel_p_np = np.asarray(kernel_p, dtype=np.float64)
    if kernel_q is None:
        kernel_q_np = kernel_p_np
    else:
        kernel_q_np = np.asarray(kernel_q, dtype=np.float64)
    n_states = int(kernel_p_np.shape[0])
    n_max = len(potentials_np) - 1
    if not 1 <= level_upper <= n_max:
        raise ValueError("level_upper must lie in 1..N")
    lower_len = n_max - (level_upper - 1) + 1
    upper_len = n_max - level_upper + 1
    lower_trajs = enumerate_trajectories(lower_len, n_states)
    upper_trajs = enumerate_trajectories(upper_len, n_states)
    lower_index = {traj: i for i, traj in enumerate(lower_trajs)}
    upper_index = {traj: j for j, traj in enumerate(upper_trajs)}
    n_lower = len(lower_trajs)
    n_upper = len(upper_trajs)
    n_pairs = n_lower * n_upper
    kernel = np.zeros((n_pairs, n_pairs), dtype=np.float64)
    g_lower = potentials_np[level_upper - 1]
    g_upper = potentials_np[level_upper]

    for i, x_lower in enumerate(lower_trajs):
        for j, x_upper in enumerate(upper_trajs):
            source = i * n_upper + j
            y_upper = x_lower[:-1]
            j_prop = upper_index[y_upper]
            end_upper = x_upper[-1]
            stay = 0.0
            for y in range(n_states):
                y_lower = x_upper + (y,)
                i_prop = lower_index[y_lower]
                dest = i_prop * n_upper + j_prop
                q_prob = float(kernel_q_np[end_upper, y])
                alpha = float(
                    acceptance_probability(
                        g_lower,
                        g_upper,
                        x_lower[-1],
                        end_upper,
                        y,
                        y_upper[-1],
                        kernel_p_np,
                        kernel_q_np,
                    )
                )
                if dest == source:
                    stay += q_prob
                else:
                    kernel[source, dest] += q_prob * alpha
                    stay += q_prob * (1.0 - alpha)
            kernel[source, source] += stay
    return kernel, lower_trajs, upper_trajs


def one_step_kernel(
    m_parity: int,
    potentials: Sequence[Array],
    kernel_p: Array,
    kernel_q: Array | None = None,
    eta: Array | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, ...]]], list[int]]:
    """One ENCORE iteration with m congruent to m_parity modulo 2.

    The state is a triple of trajectories (level 0, level 1, ..., level N).
    The index is sum_i i_level * stride_level, with the last stride equal to 1.
    Refresh of the reference level is applied before the parity's exchanges.
    """
    if m_parity not in (0, 1):
        raise ValueError("m_parity must be 0 or 1")
    potentials_np = [np.asarray(g, dtype=np.float64) for g in potentials]
    kernel_p_np = np.asarray(kernel_p, dtype=np.float64)
    if kernel_q is None:
        kernel_q_np = kernel_p_np
    else:
        kernel_q_np = np.asarray(kernel_q, dtype=np.float64)
    n_states = int(kernel_p_np.shape[0])
    if eta is None:
        eta_np = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    else:
        eta_np = np.asarray(eta, dtype=np.float64)
    n_max = len(potentials_np) - 1
    traj_lists = [
        enumerate_trajectories(n_max - level + 1, n_states) for level in range(n_max + 1)
    ]
    sizes = [len(trajs) for trajs in traj_lists]
    strides = _strides(sizes)
    n_full = strides[0] * sizes[0]
    ranges = [range(size) for size in sizes]

    refresh = np.zeros((n_full, n_full), dtype=np.float64)
    for idxs in itertools.product(*ranges):
        source = _pack(idxs, strides)
        for i_new, traj in enumerate(traj_lists[n_max]):
            updated = list(idxs)
            updated[n_max] = i_new
            refresh[source, _pack(updated, strides)] = float(eta_np[traj[0]])

    exchange = np.eye(n_full, dtype=np.float64)
    for n in range(1, n_max + 1):
        if n % 2 != m_parity:
            continue
        pair_kernel, _, _ = exchange_kernel(n, potentials_np, kernel_p_np, kernel_q_np)
        exchange = exchange @ _embed_pair(pair_kernel, n - 1, sizes, strides, n_full)
    return refresh @ exchange, traj_lists, strides


def _sample_path(key: Array, length: int, kernel: Array, eta: Array) -> Array:
    keys = jax.random.split(key, length)
    states = [int(jax.random.choice(keys[0], eta.shape[0], p=eta))]
    for step in range(1, length):
        parent = states[-1]
        states.append(int(jax.random.choice(keys[step], kernel.shape[0], p=kernel[parent])))
    return jnp.asarray(states, dtype=jnp.int32)


def _exchange(
    lower: Array,
    upper: Array,
    key: Array,
    g_lower: Array,
    g_upper: Array,
    kernel_p: Array,
    kernel_q: Array,
) -> tuple[Array, Array, float, bool]:
    key_y, key_u = jax.random.split(key)
    y_upper = truncate(lower)
    end_upper = int(upper[-1])
    y = int(jax.random.choice(key_y, kernel_q.shape[0], p=kernel_q[end_upper]))
    y_lower = extend(upper, y)
    alpha = acceptance_probability(
        g_lower,
        g_upper,
        int(lower[-1]),
        end_upper,
        y,
        int(y_upper[-1]),
        kernel_p,
        kernel_q,
    )
    u = float(jax.random.uniform(key_u, dtype=jnp.float64))
    alpha_f = float(alpha)
    if u < alpha_f:
        return y_lower, jnp.asarray(y_upper, dtype=jnp.int32), alpha_f, True
    return lower, upper, alpha_f, False


def _check_trajectory_lengths(trajectories: Sequence[Array], n_max: int) -> None:
    if len(trajectories) != n_max + 1:
        raise ValueError("expected one trajectory per level")
    for level, traj in enumerate(trajectories):
        expected = n_max - level + 1
        if int(traj.shape[0]) != expected:
            raise ValueError(f"level {level} trajectory must have length {expected}")


def _strides(sizes: Sequence[int]) -> list[int]:
    strides = [1] * len(sizes)
    for level in range(len(sizes) - 2, -1, -1):
        strides[level] = strides[level + 1] * sizes[level + 1]
    return strides


def _pack(indices: Sequence[int], strides: Sequence[int]) -> int:
    return sum(index * stride for index, stride in zip(indices, strides))


def _embed_pair(
    pair_kernel: np.ndarray,
    level_lower: int,
    sizes: Sequence[int],
    strides: Sequence[int],
    n_full: int,
) -> np.ndarray:
    n_upper = sizes[level_lower + 1]
    n_lower = sizes[level_lower]
    embedded = np.zeros((n_full, n_full), dtype=np.float64)
    ranges = [range(size) for size in sizes]
    for idxs in itertools.product(*ranges):
        source = _pack(idxs, strides)
        pair_source = idxs[level_lower] * n_upper + idxs[level_lower + 1]
        row = pair_kernel[pair_source]
        for i_lower in range(n_lower):
            base = i_lower * n_upper
            for i_upper in range(n_upper):
                prob = row[base + i_upper]
                if prob == 0.0:
                    continue
                updated = list(idxs)
                updated[level_lower] = i_lower
                updated[level_lower + 1] = i_upper
                embedded[source, _pack(updated, strides)] += prob
    return embedded
