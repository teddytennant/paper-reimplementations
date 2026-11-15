"""Learnable pieces that plug into the mechanistic solver.

Two of them, matching section 4.4 and Case B of section 4.5:

  NeuralDissolution   an MLP standing in for the hand-picked rate law, eq (15)
  LearnableSeed       the initial distribution f0 as a free variable

Both are equinox modules, so the free variables phi are just the array leaves and
anything in this file can be differentiated end to end through solver.simulate.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .grid import Grid, crystal_volume


class NeuralDissolution(eqx.Module):
    """D = NN_theta(S, T), eq (15).

    Two inputs, two outputs, tanh hidden layers. Inputs are standardised by eq
    (16) with the mean and standard deviation of the observed S and T. The output
    goes through a negative softplus so the rate is always negative, is scaled by
    a reference rate estimated as mean dL/dt over the dataset, and is clipped to
    zero for S >= 1 so nothing dissolves under supersaturation.

    Nothing here forces D -> 0 as S -> 1 from below. That is deliberate: the
    paper's central result is that the network only learns the equilibrium
    behaviour when the initial conditions are free to move too.
    """

    mlp: eqx.nn.MLP
    s_mean: jnp.ndarray
    s_std: jnp.ndarray
    t_mean: jnp.ndarray
    t_std: jnp.ndarray
    ref_rate: jnp.ndarray

    def __init__(self, key, s_mean, s_std, t_mean, t_std, ref_rate,
                 width: int = 20, depth: int = 3):
        self.mlp = eqx.nn.MLP(
            in_size=2, out_size=2, width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key,
        )
        self.s_mean = jnp.asarray(s_mean, dtype=float)
        self.s_std = jnp.asarray(s_std, dtype=float)
        self.t_mean = jnp.asarray(t_mean, dtype=float)
        self.t_std = jnp.asarray(t_std, dtype=float)
        self.ref_rate = jnp.asarray(ref_rate, dtype=float)

    def rates(self, S, T):
        """(D1, D2) for scalar S and T, before broadcasting onto the grid."""
        x = jnp.stack([(S - self.s_mean) / self.s_std, (T - self.t_mean) / self.t_std])
        y = -jax.nn.softplus(self.mlp(x)) * self.ref_rate
        return jnp.where(S < 1.0, y, 0.0)

    def __call__(self, L1, L2, S, T):
        d = self.rates(S, T)
        return jnp.broadcast_to(d[0], jnp.shape(L1)), jnp.broadcast_to(d[1], jnp.shape(L2))


class LearnableSeed(eqx.Module):
    """f0 as a free variable, rectified and renormalised to a fixed seed mass.

    Section 2.2 assimilates the whole 360 x 120 seed matrix. A ReLU keeps the
    density non-negative and the distribution is rescaled to the measured seed
    volume during simulation, so assimilation only moves the shape, not the mass.
    The paper applies the ReLU at each optimiser update; applying it inside the
    forward pass is the same fixed point and keeps the model a pure function.
    """

    raw: jnp.ndarray
    grid: Grid
    target_volume: float = eqx.field(static=True)

    def __init__(self, raw, grid: Grid, target_volume: float):
        self.raw = jnp.asarray(raw, dtype=float)
        self.grid = grid
        self.target_volume = float(target_volume)

    def __call__(self):
        f = jax.nn.relu(self.raw)
        v = crystal_volume(f, self.grid)
        return f * (self.target_volume / (v + 1e-30))


def standardise(x, mean, std):
    """Eq (16)."""
    return (x - mean) / std


class Standardiser(eqx.Module):
    """Per predictor mean and standard deviation for the loss of eq (17)."""

    mean: jnp.ndarray
    std: jnp.ndarray

    def __call__(self, x):
        return (x - self.mean) / self.std


def standardiser_from(values) -> Standardiser:
    values = jnp.asarray(values)
    std = jnp.std(values)
    return Standardiser(mean=jnp.mean(values), std=jnp.where(std > 0, std, 1.0))
