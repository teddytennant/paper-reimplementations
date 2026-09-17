"""Algorithm 1: METALICA on the 1D GMM toy."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_metalica.bias import cv_grid, deposit_hills, evaluate_V
from jax_metalica.gmm import ToyGMM, toy_gmm
from jax_metalica.mbar import free_energy_difference, mbar_weights
from jax_metalica.replica import init_chain, replica_step
from jax_metalica.sde import time_grid

Array = jax.Array


class MetalicaConfig(NamedTuple):
    """Tiny defaults for CPU tests. Paper F.1 values are noted in README."""

    M: int = 8
    N: int = 8
    K: int = 3
    rho: float = 0.5
    hill_height: float = 0.1
    hill_sigma: float = 0.2
    gamma: float = 2.0
    n_grid: int = 41
    omega_min: float = -8.0
    omega_max: float = 8.0
    eta: float = 0.0
    mbar_iters: int = 200


class MetalicaResult(NamedTuple):
    samples: Array
    weights: Array
    delta_F: Array
    V: Array
    omega: Array
    chain: Array
    V_snapshots: Array


def _advance_n_steps(
    key: Array,
    chain: Array,
    times: Array,
    omega: Array,
    V: Array,
    gmm: ToyGMM,
    rho: float,
    n_steps: int,
) -> tuple[Array, Array]:
    """N replica-exchange steps; collect the t=0 particle each step."""

    def body(carry: tuple[Array, Array], step: Array) -> tuple[tuple[Array, Array], Array]:
        k, ch = carry
        k, k_step = jax.random.split(k)
        parity = step % 2
        ch = replica_step(k_step, ch, times, omega, V, gmm, rho, parity)
        return (k, ch), ch[0]

    (key, chain), samples = jax.lax.scan(
        body, (key, chain), jnp.arange(n_steps)
    )
    return chain, samples


def run(
    key: Array,
    config: MetalicaConfig | None = None,
    gmm: ToyGMM | None = None,
) -> MetalicaResult:
    """METALICA (Algorithm 1) on the exact-score GMM toy."""
    cfg = MetalicaConfig() if config is None else config
    if gmm is None:
        gmm = toy_gmm(cfg.eta)
    times = time_grid(cfg.M)
    grid = cv_grid(cfg.n_grid, cfg.omega_min, cfg.omega_max)
    omega, V = grid.omega, grid.V
    key, k_init = jax.random.split(key)
    chain = init_chain(k_init, times, gmm)

    n_outer = cfg.K + 1
    n_samples_total = n_outer * cfg.N
    samples_all = jnp.zeros((n_samples_total,), dtype=chain.dtype)
    V_snaps = jnp.zeros((n_outer, cfg.n_grid), dtype=V.dtype)

    for k in range(n_outer):
        V_snaps = V_snaps.at[k].set(V)
        key, k_adv = jax.random.split(key)
        chain, samp = _advance_n_steps(
            k_adv, chain, times, omega, V, gmm, cfg.rho, cfg.N
        )
        samples_all = samples_all.at[k * cfg.N : (k + 1) * cfg.N].set(samp)
        V, _b = deposit_hills(
            omega,
            V,
            samp,
            h=cfg.hill_height,
            sigma=cfg.hill_sigma,
            gamma=cfg.gamma,
        )

    n_states = n_outer
    n_counts = jnp.full((n_states,), float(cfg.N))
    V_matrix = jax.vmap(lambda Vk: evaluate_V(omega, Vk, samples_all))(V_snaps)
    weights = mbar_weights(V_matrix, n_counts)
    delta_F = free_energy_difference(samples_all, weights)
    return MetalicaResult(
        samples=samples_all,
        weights=weights,
        delta_F=delta_F,
        V=V,
        omega=omega,
        chain=chain,
        V_snapshots=V_snaps,
    )
