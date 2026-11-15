"""Checks for ENCORE on the 3-state chain."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from encore import (
    acceptance_probability,
    backward_kernel,
    constant_potentials,
    enumerate_trajectories,
    exchange_kernel,
    one_step_kernel,
    reference_eta,
    run_encore,
    tilted_potentials,
)


def _masses(trajs, potential, eta, kernel):
    """Trajectory target proportional to G(endpoint) times the generative path density."""
    potential = np.asarray(potential, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    raw = np.empty(len(trajs), dtype=np.float64)
    for i, traj in enumerate(trajs):
        density = float(eta[traj[0]])
        for parent, child in zip(traj[:-1], traj[1:]):
            density *= float(kernel[parent, child])
        raw[i] = float(potential[traj[-1]]) * density
    total = raw.sum()
    assert total > 0.0
    return raw / total


def _product_target(level_pis, strides):
    sizes = [len(pi) for pi in level_pis]
    n_full = strides[0] * sizes[0]
    pi = np.zeros(n_full, dtype=np.float64)
    for idxs in itertools.product(*[range(size) for size in sizes]):
        weight = 1.0
        index = 0
        for choice, stride, level_pi in zip(idxs, strides, level_pis):
            weight *= float(level_pi[choice])
            index += choice * stride
        pi[index] = weight
    return pi


def test_constant_potentials_accept_every_exchange():
    potentials = constant_potentials()
    kernel = backward_kernel()
    eta = reference_eta()
    # Unequal constants are still constant on each level, so the ratio is 1.
    unequal = (
        jnp.full(3, 2.0, dtype=jnp.float64),
        jnp.full(3, 5.0, dtype=jnp.float64),
        jnp.full(3, 1.0, dtype=jnp.float64),
    )
    for g_lower, g_upper in ((potentials[0], potentials[1]), (unequal[0], unequal[1])):
        for ends in itertools.product(range(3), repeat=4):
            alpha = acceptance_probability(g_lower, g_upper, *ends, kernel, kernel)
            assert float(alpha) == 1.0

    result = run_encore(
        jax.random.PRNGKey(0),
        128,
        potentials,
        kernel,
        kernel,
        eta,
    )
    assert result.n_proposals == 128
    assert result.n_rejections == 0
    assert bool(jnp.all(result.alphas == 1.0))
    assert bool(jnp.all(result.accepted))


def test_acceptance_matches_formula():
    g_lower = np.array([1.0, 3.0, 1.0], dtype=np.float64)
    g_upper = np.array([1.0, 1.0, 2.0], dtype=np.float64)

    # G_l(y) * G_u(y_u) / (G_l(x) * G_u(x_u)) = (1 * 1) / (3 * 2) < 1.
    alpha_low = float(acceptance_probability(g_lower, g_upper, 1, 2, 0, 0))
    expected_low = min(1.0, (g_lower[0] * g_upper[0]) / (g_lower[1] * g_upper[2]))
    assert expected_low < 1.0
    assert alpha_low == expected_low

    # The swapped pair has ratio 6, which the Metropolis rule caps at 1.
    alpha_high = float(acceptance_probability(g_lower, g_upper, 0, 0, 1, 2))
    expected_high = min(1.0, (g_lower[1] * g_upper[2]) / (g_lower[0] * g_upper[0]))
    assert expected_high == 1.0
    assert alpha_high == 1.0

    # Both weights are 0, so the ratio is defined to be 1.
    g_zero = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    alpha_zero = float(acceptance_probability(g_zero, g_upper, 0, 1, 0, 2))
    assert alpha_zero == 1.0

    # Q != P multiplies by the density ratio P/Q from Eq. 3.2.
    kernel_p = np.array(
        [
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.125, 0.375, 0.5],
        ],
        dtype=np.float64,
    )
    kernel_q = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.5, 0.25, 0.25],
            [0.5, 0.125, 0.375],
        ],
        dtype=np.float64,
    )
    g_l = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    g_u = np.array([2.0, 1.0, 1.0], dtype=np.float64)
    x_l, x_u, y_l, y_u = 2, 0, 0, 1
    ratio = (
        g_l[y_l]
        * g_u[y_u]
        * kernel_p[x_u, y_l]
        * kernel_q[y_u, x_l]
    ) / (
        g_u[x_u]
        * g_l[x_l]
        * kernel_q[x_u, y_l]
        * kernel_p[y_u, x_l]
    )
    assert ratio < 1.0
    alpha_q = float(
        acceptance_probability(g_l, g_u, x_l, x_u, y_l, y_u, kernel_p, kernel_q)
    )
    assert alpha_q == min(1.0, ratio)

    x_l, x_u, y_l, y_u = 0, 1, 2, 0
    ratio_high = (
        g_l[y_l]
        * g_u[y_u]
        * kernel_p[x_u, y_l]
        * kernel_q[y_u, x_l]
    ) / (
        g_u[x_u]
        * g_l[x_l]
        * kernel_q[x_u, y_l]
        * kernel_p[y_u, x_l]
    )
    assert ratio_high > 1.0
    alpha_cap = float(
        acceptance_probability(g_l, g_u, x_l, x_u, y_l, y_u, kernel_p, kernel_q)
    )
    assert alpha_cap == 1.0


def test_exchange_detailed_balance_and_one_step_invariance():
    potentials = tilted_potentials()
    kernel = np.asarray(backward_kernel(), dtype=np.float64)
    eta = np.asarray(reference_eta(), dtype=np.float64)
    kernel_q = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.5, 0.25, 0.25],
            [0.125, 0.125, 0.75],
        ],
        dtype=np.float64,
    )

    # p_0 is the pushforward of eta through the two backward kernels, by
    # matrix multiplication, not by sampling. The level-0 endpoint target is
    # proportional to G_0(x) * p_0(x).
    p0 = eta @ kernel @ kernel
    pi0_endpoint = potentials[0] * p0
    pi0_endpoint = np.asarray(pi0_endpoint, dtype=np.float64)
    pi0_endpoint = pi0_endpoint / pi0_endpoint.sum()
    level0 = enumerate_trajectories(3)
    marg = np.zeros(3, dtype=np.float64)
    for traj, mass in zip(level0, _masses(level0, potentials[0], eta, kernel)):
        marg[traj[-1]] += mass
    assert np.max(np.abs(marg - pi0_endpoint)) < 1e-8

    for level_upper in (1, 2):
        for proposal in (kernel, kernel_q):
            transition, lower_trajs, upper_trajs = exchange_kernel(
                level_upper, potentials, kernel, proposal
            )
            pi_lower = _masses(lower_trajs, potentials[level_upper - 1], eta, kernel)
            pi_upper = _masses(upper_trajs, potentials[level_upper], eta, kernel)
            n_upper = len(upper_trajs)
            pi = np.empty(len(lower_trajs) * n_upper, dtype=np.float64)
            for i, left in enumerate(pi_lower):
                pi[i * n_upper : (i + 1) * n_upper] = left * pi_upper
            assert abs(pi.sum() - 1.0) < 1e-12
            assert np.max(np.abs(transition.sum(axis=1) - 1.0)) < 1e-12
            balance = pi[:, None] * transition - pi[None, :] * transition.T
            assert np.max(np.abs(balance)) < 1e-8
            assert np.max(np.abs(pi @ transition - pi)) < 1e-8

    level_trajs = [enumerate_trajectories(3 - level) for level in range(3)]
    level_pis = [
        _masses(level_trajs[level], potentials[level], eta, kernel) for level in range(3)
    ]
    for parity in (0, 1):
        transition, traj_lists, strides = one_step_kernel(
            parity, potentials, kernel, kernel, eta
        )
        assert traj_lists == level_trajs
        pi = _product_target(level_pis, strides)
        assert abs(pi.sum() - 1.0) < 1e-12
        assert np.max(np.abs(transition.sum(axis=1) - 1.0)) < 1e-12
        assert np.max(np.abs(pi @ transition - pi)) < 1e-8


def test_short_run_endpoints_in_state_space():
    result = run_encore(
        jax.random.PRNGKey(1),
        300,
        tilted_potentials(),
        backward_kernel(),
        backward_kernel(),
        reference_eta(),
    )
    endpoints = np.asarray(result.endpoints)
    assert endpoints.shape == (300,)
    assert np.all(np.isfinite(endpoints.astype(np.float64)))
    assert set(np.unique(endpoints)).issubset({0, 1, 2})
