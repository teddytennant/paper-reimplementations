"""Uniform finite volume grid over the two characteristic lengths L = (L1, L2).

The PBE of eq (1) lives on a two dimensional internal coordinate: L1 is the needle
length and L2 the width, both in micrometres. Cells are uniform in both directions,
matching the 360 x 120 grid quoted in section 2.2 of the paper.

The number density f is stored as cell averages with shape (n1, n2), in units of
number per kg of solvent per um^2.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class Grid(eqx.Module):
    """Uniform cell-centred grid on [0, L1_max] x [0, L2_max].

    Every field is static, so a Grid is an empty pytree and can be passed straight
    through jit without being traced.
    """

    n1: int = eqx.field(static=True)
    n2: int = eqx.field(static=True)
    L1_max: float = eqx.field(static=True)
    L2_max: float = eqx.field(static=True)

    @property
    def dL1(self) -> float:
        return self.L1_max / self.n1

    @property
    def dL2(self) -> float:
        return self.L2_max / self.n2

    @property
    def cell_area(self) -> float:
        return self.dL1 * self.dL2

    @property
    def L1_edges(self):
        return jnp.linspace(0.0, self.L1_max, self.n1 + 1)

    @property
    def L2_edges(self):
        return jnp.linspace(0.0, self.L2_max, self.n2 + 1)

    @property
    def L1_centers(self):
        return self.L1_edges[:-1] + 0.5 * self.dL1

    @property
    def L2_centers(self):
        return self.L2_edges[:-1] + 0.5 * self.dL2

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n1, self.n2)

    def mesh_centers(self):
        """(L1, L2) at cell centres, both shape (n1, n2)."""
        return (self.L1_centers[:, None] + 0.0 * self.L2_centers[None, :],
                0.0 * self.L1_centers[:, None] + self.L2_centers[None, :])


def moment(f, grid: Grid, p: int = 0, q: int = 0):
    """Cross moment mu_pq = int L1^p L2^q f dL1 dL2, by midpoint rule."""
    w = (grid.L1_centers[:, None] ** p) * (grid.L2_centers[None, :] ** q)
    return jnp.sum(f * w) * grid.cell_area


def crystal_volume(f, grid: Grid):
    """Total crystal volume Vc [um^3 kg^-1] without the shape factor.

    Eq (2) carries kv = pi/4 outside the integral, so Vc here is mu_12 =
    int L1 L2^2 f dL, the volume of a cylinder of length L1 and diameter L2
    divided by kv.
    """
    return moment(f, grid, 1, 2)


def mean_lengths(f, grid: Grid, eps: float = 1e-30):
    """Volume weighted mean length and width (L1bar, L2bar).

    These are two of the three training targets in section 4.5. Weighting by
    crystal volume L1 L2^2 gives L1bar = mu_22 / mu_12 and L2bar = mu_13 / mu_12.
    """
    v = moment(f, grid, 1, 2)
    l1 = moment(f, grid, 2, 2) / (v + eps)
    l2 = moment(f, grid, 1, 3) / (v + eps)
    return l1, l2


def aspect_ratio(f, grid: Grid):
    """L1bar / L2bar, the product specification targeted in section 2.3.2."""
    l1, l2 = mean_lengths(f, grid)
    return l1 / l2
