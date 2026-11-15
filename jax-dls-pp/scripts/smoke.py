"""8-node synthetic inhomogeneous Poisson network; short SGD fit."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from dls_pp.estimate import fit_dls_pp, linear_coefficients, simulate_network
from dls_pp.likelihood import dyad_indices
from dls_pp.splines import evaluate_trajectories, make_basis


def main() -> None:
    n = 8
    t_max = 1.0
    n_coeff = 5
    basis = make_basis(t_max, n_coeff)

    # Two drifting clusters of four nodes on [0, 1].
    intercept = jnp.array(
        [
            [-0.7, 0.05],
            [-0.65, -0.08],
            [-0.75, 0.04],
            [-0.68, -0.02],
            [0.7, 0.06],
            [0.66, -0.05],
            [0.74, 0.03],
            [0.69, -0.04],
        ]
    )
    slope = jnp.array(
        [
            [0.25, 0.15],
            [0.22, 0.12],
            [0.28, 0.10],
            [0.20, 0.18],
            [-0.22, -0.12],
            [-0.25, -0.10],
            [-0.18, -0.15],
            [-0.24, -0.08],
        ]
    )
    beta = jnp.array([0.4, 0.5, 0.35, 0.45, 0.4, 0.55, 0.3, 0.5])
    c_x, c_y = linear_coefficients(intercept, slope, basis.knots)

    key = random.PRNGKey(11)
    events = simulate_network(key, beta, c_x, c_y, basis.knots, t_max, model="full")

    result = fit_dls_pp(
        events,
        n_coeff=n_coeff,
        n_steps=200,
        batch_size=16,
        lr=0.06,
        tau=1.0,
        rho1=0.08,
        rho2=0.02,
        n_quad=16,
        model="full",
        seed=2,
        beta_every=10,
    )
    ll0 = float(result.history[0])
    ll1 = float(result.history[-1])
    print(f"train loglik before: {ll0:.6f}")
    print(f"train loglik after:  {ll1:.6f}")
    print(f"loglik increased: {ll1 > ll0}")
    print(f"n_events: {events.n_events}")

    times = jnp.linspace(0.0, t_max, 9)
    traj = evaluate_trajectories(result.c_x, result.c_y, times, basis.knots)
    z_bar = jnp.mean(traj, axis=1)
    src, dst = dyad_indices(n)
    d2 = jnp.sum((z_bar[src] - z_bar[dst]) ** 2, axis=-1)

    counts = jnp.zeros((n, n), dtype=jnp.float64)
    if events.n_events > 0:
        counts = counts.at[events.src, events.dst].add(1.0)
        counts = counts + counts.T
    rates = counts[src, dst]
    median = jnp.median(rates)
    high = rates >= median
    low = ~high
    high_dist = float(jnp.mean(jnp.sqrt(d2[high]))) if bool(jnp.any(high)) else float("nan")
    low_dist = float(jnp.mean(jnp.sqrt(d2[low]))) if bool(jnp.any(low)) else float("nan")
    print(f"mean pairwise distance (high-rate dyads): {high_dist:.6f}")
    print(f"mean pairwise distance (low-rate dyads):  {low_dist:.6f}")


if __name__ == "__main__":
    main()
