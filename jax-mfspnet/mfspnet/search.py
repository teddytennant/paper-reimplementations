"""The MFSPNet search loop (paper Alg. 3) and a random-search baseline.

Alg. 3 initializes the swarm with Alg. 2, then for each generation updates
every particle with Eqs. 4 and 5, scores it by training the decoded block for c
cutoff epochs and taking VLE-EMA (Eq. 8), keeps a personal best only when the
new position beats it, and takes the global best over personal bests.

Table 2 and Table 3 give the search-time settings: 20 particles, 55
generations, c = 15 epochs, 8x8 proxy images, batch 256, Adam at 1e-3. That is
about 1100 trained candidates. :class:`SearchConfig` defaults to the paper's
numbers; the toy runs in the tests override them down to something a CPU can
finish, which is a change of scale and nothing else.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np

from .data import Dataset
from .encoding import SearchSpace, decode, initialize_population, random_particle
from .estimators import CUTOFF_EPOCHS, GAMMA, vle_ema
from .model import StackConfig
from .pso import PSOConfig, minimize
from .train import train_candidate


@dataclasses.dataclass
class SearchConfig:
    """Search-time settings. Defaults are Table 2 and Table 3."""

    n_particles: int = 20
    n_generations: int = 55
    cutoff_epochs: int = CUTOFF_EPOCHS
    gamma: float = GAMMA
    batch_size: int = 256
    stem_channels: int = 16
    n_blocks: int = 1


def architecture_id(growth_rates) -> int:
    """A stable integer id for a decoded block, used to fix its weight init.

    Python's builtin ``hash`` is fine for int tuples but this spells the mixing
    out so the id does not move with the interpreter.
    """
    h = 2166136261
    for g in growth_rates:
        h = ((h ^ (int(g) + 1)) * 16777619) & 0xFFFFFFFF
    return int(h % (2**31 - 1))


@dataclasses.dataclass
class Evaluation:
    particle: np.ndarray
    growth_rates: list
    fitness: float
    val_accuracy: float


class SurrogateEvaluator:
    """Alg. 3 line 10: train for c epochs, score with VLE-EMA.

    Results are cached on the decoded (integer) particle, so a swarm that
    revisits a lattice point does not pay for it twice. The weight-init key is
    folded from the decoded architecture rather than drawn fresh, so the
    fitness is a deterministic function of the architecture instead of a
    function of the architecture and a draw. Neither detail is in the paper.
    Caching changes cost only. The fixed init makes two candidates comparable
    under the same initialization, which matters at the tiny scale used here,
    where init noise is the same size as the gap between architectures.
    """

    def __init__(self, key, data: Dataset, space: SearchSpace, cfg: SearchConfig):
        self.base_key = key
        self.data = data
        self.space = space
        self.cfg = cfg
        self.cache: dict[tuple, Evaluation] = {}
        self.history: list[Evaluation] = []

    @property
    def n_trained(self) -> int:
        return len(self.cache)

    def evaluate_one(self, particle: np.ndarray) -> Evaluation:
        rates = decode(particle, self.space)
        key_t = tuple(rates)
        if key_t in self.cache:
            hit = self.cache[key_t]
            self.history.append(hit)
            return hit
        model_cfg = StackConfig(
            n_classes=self.data.n_classes,
            stem_channels=self.cfg.stem_channels,
            n_blocks=self.cfg.n_blocks,
        )
        sub = jax.random.fold_in(self.base_key, architecture_id(key_t))
        traj = train_candidate(
            sub,
            rates,
            self.data,
            model_cfg,
            epochs=self.cfg.cutoff_epochs,
            batch_size=self.cfg.batch_size,
        )
        result = Evaluation(
            particle=np.asarray(particle).copy(),
            growth_rates=rates,
            fitness=vle_ema(traj.val_losses, self.cfg.gamma),
            val_accuracy=traj.val_accuracy,
        )
        self.cache[key_t] = result
        self.history.append(result)
        return result

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray([self.evaluate_one(p).fitness for p in positions])


@dataclasses.dataclass
class SearchResult:
    growth_rates: list
    particle: np.ndarray
    fitness: float
    history: np.ndarray
    evaluations: list
    n_trained: int


def search(
    key,
    data: Dataset,
    space: SearchSpace,
    cfg: SearchConfig | None = None,
    pso_cfg: PSOConfig | None = None,
    seed: int = 0,
) -> SearchResult:
    """Alg. 3 end to end. Returns the evolved block and the convergence curve."""
    cfg = cfg or SearchConfig()
    rng = np.random.default_rng(seed)
    evaluator = SurrogateEvaluator(key, data, space, cfg)
    positions = initialize_population(rng, cfg.n_particles, space)
    state, history = minimize(
        evaluator,
        space,
        cfg.n_particles,
        cfg.n_generations,
        rng,
        pso_cfg,
        positions=positions,
    )
    return SearchResult(
        growth_rates=decode(state.gbest, space),
        particle=state.gbest,
        fitness=state.gbest_fitness,
        history=history,
        evaluations=list(evaluator.history),
        n_trained=evaluator.n_trained,
    )


def random_search(
    key,
    data: Dataset,
    space: SearchSpace,
    n_candidates: int,
    cfg: SearchConfig | None = None,
    seed: int = 0,
    max_draws: int | None = None,
) -> SearchResult:
    """Same fitness, particles drawn with Alg. 1 instead of evolved.

    ``n_candidates`` counts *distinct trained* architectures, so calling this
    with ``len(pso.evaluations)`` would be unfair; pass the PSO run's
    ``n_trained`` to match training cost. This control is not in the paper; the
    paper's ablation compares against PSO without the surrogate (Sec. 6.8).
    """
    cfg = cfg or SearchConfig()
    rng = np.random.default_rng(seed)
    evaluator = SurrogateEvaluator(key, data, space, cfg)
    best = None
    history = []
    budget = max_draws if max_draws is not None else 50 * n_candidates
    while evaluator.n_trained < n_candidates:
        if len(history) >= budget:
            raise RuntimeError(
                f"drew {budget} particles and only reached {evaluator.n_trained} "
                f"distinct architectures out of {n_candidates}; the space is too small"
            )
        result = evaluator.evaluate_one(random_particle(rng, space).astype(np.float64))
        if best is None or result.fitness < best.fitness:
            best = result
        history.append(best.fitness)
    return SearchResult(
        growth_rates=best.growth_rates,
        particle=best.particle,
        fitness=best.fitness,
        history=np.asarray(history),
        evaluations=list(evaluator.history),
        n_trained=evaluator.n_trained,
    )
