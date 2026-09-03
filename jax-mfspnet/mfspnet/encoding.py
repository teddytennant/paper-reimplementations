"""Variable-length particle encoding for dense blocks (paper Sec. 4.1, Alg. 1)
and the LHS + standard-deviation population initializer (Sec. 4.2, Alg. 2).

The encoding is a fixed-length integer vector of length ``l_max``. Cell ``i``
holds the growth rate of composite layer ``i``, or the special flag
``S_v = r_lower - 1`` if that layer is switched off. Variable block length comes
out of the flag, not out of variable-length vectors, which is what lets the
paper reuse textbook PSO on the encoding (Sec. 4.5, first paragraph).

Paper defaults: ``l_max = 16``, ``[r_lower, r_upper] = [12, 32]``, so
``S_v = 11`` (Sec. 4.1 and Table 2).
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SearchSpace:
    """Bounds of the dense-block search space.

    ``l_max``, ``r_lower`` and ``r_upper`` are the three hyperparameters listed
    in Sec. 4.1. The defaults are the paper's. Tests and the toy search use
    smaller spaces so a CPU can run them, which is a scale change, not a
    change of the scheme.
    """

    l_max: int = 16
    r_lower: int = 12
    r_upper: int = 32

    def __post_init__(self) -> None:
        if self.r_lower < 1:
            raise ValueError("r_lower must be at least 1 so S_v stays positive")
        if self.r_upper < self.r_lower:
            raise ValueError("r_upper must be >= r_lower")
        if self.l_max < 1:
            raise ValueError("l_max must be at least 1")

    @property
    def flag_value(self) -> int:
        """The disabled-layer flag ``S_v = r_lower - 1`` (Sec. 4.1)."""
        return self.r_lower - 1

    @property
    def lower_bound(self) -> float:
        """Smallest legal cell value; the flag itself."""
        return float(self.flag_value)

    @property
    def upper_bound(self) -> float:
        return float(self.r_upper)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clamp continuous PSO positions into the encodable interval."""
        return np.clip(x, self.lower_bound, self.upper_bound)


def encode(growth_rates, space: SearchSpace) -> np.ndarray:
    """Build a particle vector from the growth rates of the active layers.

    ``growth_rates`` is the decoded form: one entry per *active* composite
    layer. Inactive layers are appended as ``S_v`` so the vector is always
    ``l_max`` long. This is the inverse of :func:`decode` up to the position of
    the disabled cells, so :func:`decode` returns the mask as well when asked.
    """
    rates = [int(g) for g in growth_rates]
    if len(rates) > space.l_max:
        raise ValueError(f"{len(rates)} active layers exceeds l_max={space.l_max}")
    for g in rates:
        if not space.r_lower <= g <= space.r_upper:
            raise ValueError(f"growth rate {g} outside [{space.r_lower}, {space.r_upper}]")
    pad = [space.flag_value] * (space.l_max - len(rates))
    return np.asarray(rates + pad, dtype=np.int64)


def encode_with_mask(growth_rates, mask, space: SearchSpace) -> np.ndarray:
    """Encode into given slots. ``mask[i]`` true means layer ``i`` is active.

    Alg. 1 draws the flag per slot first, then fills a growth rate where the
    flag is 1, so the active layers do not have to be contiguous. The worked
    example in Sec. 4.1 is ``[16, 11, 28, 20, 11, 24, 11, 12]``.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (space.l_max,):
        raise ValueError(f"mask must have shape ({space.l_max},)")
    rates = [int(g) for g in growth_rates]
    if len(rates) != int(mask.sum()):
        raise ValueError("number of growth rates must equal the number of active slots")
    out = np.full(space.l_max, space.flag_value, dtype=np.int64)
    it = iter(rates)
    for i in range(space.l_max):
        if mask[i]:
            g = next(it)
            if not space.r_lower <= g <= space.r_upper:
                raise ValueError(f"growth rate {g} outside [{space.r_lower}, {space.r_upper}]")
            out[i] = g
    return out


def decode(particle, space: SearchSpace, return_mask: bool = False):
    """Read a particle vector back as a list of active growth rates.

    A cell counts as active when its rounded value is at least ``r_lower``.
    Continuous PSO positions are rounded here, which is where the discrete
    architecture comes from. Cells below ``r_lower`` are the flag ``S_v``.
    """
    x = np.rint(np.asarray(particle, dtype=np.float64)).astype(np.int64)
    x = np.clip(x, space.flag_value, space.r_upper)
    if x.shape != (space.l_max,):
        raise ValueError(f"particle must have shape ({space.l_max},)")
    mask = x >= space.r_lower
    rates = [int(v) for v in x[mask]]
    if return_mask:
        return rates, mask
    return rates


def canonicalize(particle, space: SearchSpace) -> np.ndarray:
    """Snap a continuous position onto the integer lattice of legal vectors."""
    rates, mask = decode(particle, space, return_mask=True)
    return encode_with_mask(rates, mask, space)


def random_particle(rng: np.random.Generator, space: SearchSpace) -> np.ndarray:
    """Alg. 1: per-slot coin flip, then a uniform growth rate where enabled."""
    out = np.full(space.l_max, space.flag_value, dtype=np.int64)
    for i in range(space.l_max):
        if rng.integers(0, 2) == 1:
            out[i] = int(rng.integers(space.r_lower, space.r_upper + 1))
    return out


def latin_hypercube(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Plain LHS on the unit cube: one stratified sample per row per dimension."""
    cut = np.arange(n)[:, None] + rng.random((n, d))
    samples = cut / n
    for j in range(d):
        rng.shuffle(samples[:, j])
    return samples


def initialize_population(
    rng: np.random.Generator,
    n_particles: int,
    space: SearchSpace,
    return_samples: bool = False,
):
    """Alg. 2: LHS samples, then sort by per-sample standard deviation.

    The unit-cube samples are mapped onto ``[S_v, r_upper]`` and rounded, so a
    coordinate that lands below ``r_lower`` becomes a disabled layer. The sort
    is descending in the spread of a particle's own coordinates, which is what
    the algorithm asks for; with a fixed swarm size the sort only reorders the
    population, so it changes which particle is index 0, not the set.

    ``return_samples`` also hands back the pre-rounding unit-cube samples, which
    is what the sort is defined on.

    Alg. 2 says nothing about the flag, so the mapping onto [S_v, r_upper] is
    an interpretation. It has a consequence worth knowing: only the bottom half
    unit of a 21-wide interval rounds to S_v, so an LHS-initialized swarm
    starts with roughly 2% of its layers disabled, where Alg. 1's per-slot coin
    flip would disable half of them. The swarm reaches short blocks through the
    velocity update rather than at initialization.
    """
    d = space.l_max
    samples = latin_hypercube(rng, n_particles, d)
    sd = samples.std(axis=1, ddof=1) if d > 1 else np.zeros(n_particles)
    order = np.argsort(-sd, kind="stable")
    samples = samples[order]
    scaled = space.lower_bound + samples * (space.upper_bound - space.lower_bound)
    population = np.rint(scaled)
    if return_samples:
        return population, samples
    return population
