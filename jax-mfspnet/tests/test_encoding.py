"""Encoding round trips and search-space coverage (paper Sec. 4.1, Alg. 1)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfspnet.encoding import (
    SearchSpace,
    canonicalize,
    decode,
    encode,
    encode_with_mask,
    initialize_population,
    latin_hypercube,
    random_particle,
)
from mfspnet.model import StackConfig, apply_network, init_network

PAPER = SearchSpace()
TOY = SearchSpace(l_max=3, r_lower=2, r_upper=3)


def test_paper_defaults():
    assert (PAPER.l_max, PAPER.r_lower, PAPER.r_upper) == (16, 12, 32)
    assert PAPER.flag_value == 11  # S_v = r_lower - 1, Sec. 4.1


def test_worked_example_from_section_4_1():
    """P = [16, 11, 28, 20, 11, 24, 11, 12] with layers 2, 5, 7 disabled."""
    space = SearchSpace(l_max=8, r_lower=12, r_upper=32)
    mask = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    p = encode_with_mask([16, 28, 20, 24, 12], mask, space)
    assert p.tolist() == [16, 11, 28, 20, 11, 24, 11, 12]
    rates, back = decode(p, space, return_mask=True)
    assert rates == [16, 28, 20, 24, 12]
    assert np.array_equal(back, mask)


@pytest.mark.parametrize("space", [PAPER, TOY])
def test_encode_decode_encode_is_identity(space):
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = random_particle(rng, space)
        rates, mask = decode(p, space, return_mask=True)
        assert np.array_equal(encode_with_mask(rates, mask, space), p)


def test_decode_encode_decode_is_identity_on_contiguous_form():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(0, PAPER.l_max + 1))
        rates = [int(g) for g in rng.integers(PAPER.r_lower, PAPER.r_upper + 1, n)]
        assert decode(encode(rates, PAPER), PAPER) == rates


def test_canonicalize_snaps_continuous_positions_onto_the_lattice():
    space = TOY
    x = np.array([1.4, 2.6, 2.49])
    p = canonicalize(x, space)
    assert p.tolist() == [1, 3, 2]
    assert np.array_equal(canonicalize(p, space), p)


def test_out_of_range_positions_are_clipped_not_rejected():
    rates, mask = decode(np.array([-50.0, 100.0, 3.0]), TOY, return_mask=True)
    assert rates == [3, 3]
    assert mask.tolist() == [False, True, True]


def test_every_toy_architecture_builds_and_runs():
    """Exhaustive over the toy space: every decodable vector gives a network."""
    values = [TOY.flag_value] + list(range(TOY.r_lower, TOY.r_upper + 1))
    x = jnp.zeros((2, 4, 4, 3))
    cfg = StackConfig(n_classes=4, stem_channels=4, n_blocks=1)
    seen = set()
    for combo in itertools.product(values, repeat=TOY.l_max):
        rates = tuple(decode(np.array(combo), TOY))
        if rates in seen:
            continue
        seen.add(rates)
        params = init_network(jax.random.key(0), list(rates), 3, cfg)
        assert apply_network(params, x, cfg).shape == (2, 4)
    assert len(seen) == 1 + 2 + 4 + 8  # 0, 1, 2 or 3 active layers


def test_all_disabled_block_is_the_identity_path():
    p = np.full(PAPER.l_max, PAPER.flag_value)
    assert decode(p, PAPER) == []
    cfg = StackConfig(n_classes=3, stem_channels=5, n_blocks=1)
    params = init_network(jax.random.key(0), [], 3, cfg)
    assert apply_network(params, jnp.zeros((2, 8, 8, 3)), cfg).shape == (2, 3)


def test_widest_paper_architecture_runs():
    """16 layers at growth 32, the corner of the paper's own search space."""
    rates = [PAPER.r_upper] * PAPER.l_max
    cfg = StackConfig(n_classes=10, stem_channels=16, n_blocks=1)
    params = init_network(jax.random.key(0), rates, 3, cfg)
    out = apply_network(params, jnp.zeros((2, 8, 8, 3)), cfg)
    assert out.shape == (2, 10)
    assert np.isfinite(np.asarray(out)).all()


def test_latin_hypercube_is_stratified():
    rng = np.random.default_rng(0)
    n, d = 20, 6
    s = latin_hypercube(rng, n, d)
    assert s.shape == (n, d)
    assert (s >= 0).all() and (s < 1).all()
    for j in range(d):
        strata = np.floor(s[:, j] * n).astype(int)
        assert sorted(strata) == list(range(n))


def test_population_init_is_sorted_by_descending_spread():
    """Alg. 2 lines 7-9: descending sample standard deviation."""
    rng = np.random.default_rng(3)
    pop, samples = initialize_population(rng, 12, PAPER, return_samples=True)
    assert pop.shape == (12, PAPER.l_max)
    sd = samples.std(axis=1, ddof=1)
    assert np.all(np.diff(sd) <= 1e-12)
    assert pop.min() >= PAPER.lower_bound and pop.max() <= PAPER.upper_bound


def test_population_init_covers_both_enabled_and_disabled_layers():
    rng = np.random.default_rng(4)
    pop = initialize_population(rng, 20, PAPER)
    lengths = [len(decode(p, PAPER)) for p in pop]
    assert min(lengths) < max(lengths)  # variable length is actually reached
