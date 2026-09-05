"""Synthetic stand-in for the Botschi et al. dissolution dataset.

The 13 real experimental trajectories of the paper are not redistributed here, so
this module generates the same three predictors (c, L1bar, L2bar) by running the
solver forward with a known published law, M1. Each synthetic experiment has its
own temperature ramp, seed distribution and starting concentration, and the seed
handed to a training run is perturbed away from the truth to stand in for the
initial condition measurement error that section 2.2 diagnoses.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .grid import Grid, crystal_volume
from .kinetics import concentration_from_S
from .solver import TemperatureProfile, simulate


def lognormal_seed(grid: Grid, median1, sigma1, median2, sigma2, total_volume=1.0e12):
    """Bivariate lognormal seed distribution, normalised to a crystal volume.

    Real seeds of needle-like L-glutamic acid are roughly lognormal in each
    dimension. total_volume is Vc = int L1 L2^2 f dL in um^3 per kg of solvent.
    """
    l1 = grid.L1_centers[:, None]
    l2 = grid.L2_centers[None, :]
    z1 = (jnp.log(l1) - jnp.log(median1)) / sigma1
    z2 = (jnp.log(l2) - jnp.log(median2)) / sigma2
    f = jnp.exp(-0.5 * (z1**2 + z2**2)) / (l1 * l2)
    return f * (total_volume / crystal_volume(f, grid))


def gaussian_seed(grid: Grid, mu1, sigma1, mu2, sigma2, total_volume=1.0e12):
    """Bivariate Gaussian seed. Smooth enough to compare against exact advection."""
    l1 = grid.L1_centers[:, None]
    l2 = grid.L2_centers[None, :]
    f = jnp.exp(-0.5 * (((l1 - mu1) / sigma1) ** 2 + ((l2 - mu2) / sigma2) ** 2))
    return f * (total_volume / crystal_volume(f, grid))


class Experiment(eqx.Module):
    """One synthetic batch: the operating condition, the truth, and what is seen."""

    t: jnp.ndarray
    T_values: jnp.ndarray
    f0_true: jnp.ndarray
    c0_true: jnp.ndarray
    f0_measured: jnp.ndarray
    c0_measured: jnp.ndarray
    c_obs: jnp.ndarray
    L1_obs: jnp.ndarray
    L2_obs: jnp.ndarray

    @property
    def temperature(self):
        return TemperatureProfile(times=self.t, values=self.T_values)


class Dataset(NamedTuple):
    experiments: list
    grid: Grid
    seed_volume: float


# A fixed experimental design, five batches spanning the temperatures and
# starting supersaturations of the real dataset. Held fixed across runs so an
# ensemble of models differs only in its random initialisation, the way the 20
# member ensembles of Figure 2(a) do.
DEFAULT_DESIGN = (
    # (T at t=0, T at t=end, S at t=0, seed median L1, seed median L2)
    (26.0, 28.0, 0.9780, 300.0, 55.0),
    (28.0, 30.5, 0.9850, 340.0, 60.0),
    (30.0, 31.2, 0.9920, 380.0, 65.0),
    (24.0, 27.0, 0.9760, 320.0, 58.0),
    (32.0, 33.5, 0.9900, 360.0, 62.0),
)


def make_dataset(key=None, grid: Grid = None, kinetics=None, design=DEFAULT_DESIGN,
                 duration: float = 3600.0, n_obs: int = 21, n_substeps: int = 6,
                 noise: float = 0.0, ic_shift: float = 0.06,
                 seed_volume: float = 1.0e12) -> Dataset:
    """Run the design forward with a known kinetic law and record the predictors.

    key only drives the observation noise, so with noise = 0 the dataset is
    deterministic. ic_shift is the relative error deliberately baked into the
    measured seed, the thing data assimilation has to undo.
    """
    t = jnp.linspace(0.0, duration, n_obs)
    if key is None:
        key = jax.random.PRNGKey(0)
    experiments = []
    for T0, T1, S0, med1, med2 in design:
        key, k_noise = jax.random.split(key)
        T_values = jnp.linspace(T0, T1, n_obs)
        f0 = lognormal_seed(grid, med1, 0.30, med2, 0.25, seed_volume)
        c0 = concentration_from_S(S0, T0)

        temperature = TemperatureProfile(times=t, values=T_values)
        res = simulate(f0, c0, temperature, t, kinetics, grid, n_substeps=n_substeps)

        kc, k1, k2 = jax.random.split(k_noise, 3)
        c_obs = res.c * (1.0 + noise * jax.random.normal(kc, res.c.shape))
        L1_obs = res.L1 * (1.0 + noise * jax.random.normal(k1, res.L1.shape))
        L2_obs = res.L2 * (1.0 + noise * jax.random.normal(k2, res.L2.shape))

        # The "measured" seed is the truth stretched along L1, the failure mode
        # the paper diagnoses in the DISCO initial conditions.
        f0_meas = lognormal_seed(grid, med1 * (1.0 + ic_shift), 0.30, med2, 0.25,
                                 seed_volume)

        experiments.append(Experiment(
            t=t, T_values=T_values, f0_true=f0, c0_true=c0,
            f0_measured=f0_meas, c0_measured=c0,
            c_obs=c_obs, L1_obs=L1_obs, L2_obs=L2_obs,
        ))
    return Dataset(experiments=experiments, grid=grid, seed_volume=seed_volume)
