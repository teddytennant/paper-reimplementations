"""Alg. 3 end to end at toy scale, and against random search.

Everything here runs on synthetic 8x8 images with a 4-slot search space and a
handful of generations. The paper's own setting is 20 particles, 55
generations, 15 cutoff epochs and 1100 trained candidates on downscaled
CIFAR-10 (Table 3), which is three and a half hours on a P40.
"""

import jax
import numpy as np
import pytest

from mfspnet.data import synthetic_task
from mfspnet.encoding import SearchSpace, decode
from mfspnet.search import (
    SearchConfig,
    SurrogateEvaluator,
    architecture_id,
    random_search,
    search,
)

TOY_SPACE = SearchSpace(l_max=4, r_lower=2, r_upper=4)
SEED = 0
TOY_CFG = SearchConfig(
    n_particles=4,
    n_generations=6,
    cutoff_epochs=4,
    batch_size=64,
    stem_channels=6,
    n_blocks=1,
)


@pytest.fixture(scope="module")
def toy_data():
    return synthetic_task(jax.random.key(0), n_train=128, n_val=64)


@pytest.fixture(scope="module")
def evolved(toy_data):
    """One toy search, shared by the tests below. Each fresh search pays for a
    fresh XLA compile per architecture, so running it once matters."""
    return search(jax.random.key(1), toy_data, TOY_SPACE, TOY_CFG, seed=SEED)


def test_paper_search_defaults():
    """Table 2 and Table 3."""
    cfg = SearchConfig()
    assert cfg.n_particles == 20
    assert cfg.n_generations == 55
    assert cfg.cutoff_epochs == 15
    assert cfg.gamma == 0.78
    assert cfg.batch_size == 256


def test_architecture_id_is_stable_and_distinguishes_blocks():
    assert architecture_id([4, 5]) == architecture_id([4, 5])
    assert architecture_id([4, 5]) != architecture_id([5, 4])
    assert architecture_id([4, 5]) != architecture_id([4, 5, 5])
    assert architecture_id([]) == architecture_id([])


def test_surrogate_fitness_is_a_function_of_the_decoded_architecture(toy_data):
    """Two particles that decode to the same block must score the same, and
    the second one must not cost anything to score."""
    ev = SurrogateEvaluator(jax.random.key(1), toy_data, TOY_SPACE, TOY_CFG)
    a = ev.evaluate_one(np.array([4.0, 1.0, 3.0, 1.0]))  # flag value is 1 here
    b = ev.evaluate_one(np.array([1.0, 4.0, 1.0, 3.0]))
    assert a.growth_rates == b.growth_rates == [4, 3]
    assert a.fitness == b.fitness
    assert ev.n_trained == 1


def test_search_returns_a_decodable_block_and_a_monotone_convergence_curve(evolved):
    result = evolved
    assert 0 < len(result.growth_rates) <= TOY_SPACE.l_max
    assert decode(result.particle, TOY_SPACE) == result.growth_rates
    assert np.all(np.diff(result.history) <= 1e-12)  # Eq. 3, gbest never worsens
    assert result.history[-1] == pytest.approx(result.fitness)
    assert len(result.history) == TOY_CFG.n_generations + 1
    assert result.n_trained <= len(result.evaluations)


def test_search_beats_random_search_on_equal_training_budget(toy_data, evolved):
    """One fixed seed at toy scale. Both runs train the same number of
    distinct architectures and score them with the same VLE-EMA, so the only
    difference is where the candidates come from. This is a single-seed check,
    not a claim about the expected margin.
    """
    sampled = random_search(
        jax.random.key(1), toy_data, TOY_SPACE, evolved.n_trained, TOY_CFG, seed=SEED
    )
    assert sampled.n_trained == evolved.n_trained
    assert evolved.fitness < sampled.fitness


def test_zero_generations_is_just_the_initial_population(toy_data):
    cfg = SearchConfig(**{**TOY_CFG.__dict__, "n_generations": 0})
    result = search(jax.random.key(1), toy_data, TOY_SPACE, cfg, seed=SEED)
    assert result.history.size == 1
    assert result.n_trained <= cfg.n_particles
