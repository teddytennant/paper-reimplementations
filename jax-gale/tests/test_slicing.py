"""The correctness anchor of Section 3.1: with the full receptive field as
overlap, sliced inference is not an approximation at all."""

import jax
import jax.numpy as jnp
import pytest
from _helpers import conv_stack, gate_stack, noise, pool_stack

from gale import (
    bands,
    exact_forward,
    make_stack,
    overlap_profile,
    receptive_field,
    sliced_forward,
    suffix_receptive_fields,
)


def stacks():
    return {"conv": conv_stack(0), "pool": pool_stack(1)}


@pytest.mark.parametrize("name", ["conv", "pool"])
@pytest.mark.parametrize("n_slices", [2, 3, 5])
def test_full_overlap_is_bit_exact(name, n_slices):
    """O_t = R_t(L) at every tensor reproduces whole-map inference.

    This runs in float64 with HIGHEST matmul precision, so the tolerance is a
    handful of ulps rather than a TF32 alibi.
    """
    ops = stacks()[name]
    x = noise(3, (1, 36, 10, 2))
    ref = exact_forward(ops, x)
    y, _ = sliced_forward(ops, x, n_slices, suffix_receptive_fields(ops))
    assert y.shape == ref.shape
    assert float(jnp.max(jnp.abs(y - ref))) < 1e-13


@pytest.mark.parametrize("n_slices", [2, 4])
def test_propagated_overlap_is_also_exact_and_smaller_than_ppbi(n_slices):
    """The decaying profile of Section 3.1 hits exactness with less overlap than
    the constant R_0(L) halo PPBI carries at every layer."""
    ops = conv_stack(0)
    x = noise(4, (1, 36, 10, 2))
    ref = exact_forward(ops, x)
    halos = overlap_profile(ops, x.shape, 4)
    y, _ = sliced_forward(ops, x, n_slices, halos)
    assert float(jnp.max(jnp.abs(y - ref))) < 1e-13
    assert list(halos) == sorted(halos, reverse=True)
    assert sum(halos) < len(halos) * halos[0]


def test_make_stack_builds_something_the_sliced_path_reproduces():
    """The convenience constructor, on a stack with a stride in it."""
    ops = make_stack(jax.random.PRNGKey(0), 2, (4, 6), (3, 3), (1, 2))
    x = noise(8, (1, 32, 8, 2))
    y, _ = sliced_forward(ops, x, 4, suffix_receptive_fields(ops))
    assert float(jnp.max(jnp.abs(y - exact_forward(ops, x)))) < 1e-13


def test_single_slice_is_the_whole_map_path():
    ops = conv_stack(2)
    x = noise(5, (1, 24, 8, 2))
    y, _ = sliced_forward(ops, x, 1)
    assert float(jnp.max(jnp.abs(y - exact_forward(ops, x)))) == 0.0


@pytest.mark.parametrize("n_slices", [1, 2, 3, 7])
def test_bands_tile_every_tensor_without_gap_or_repeat(n_slices):
    """Merging is only well defined if the bands partition each tensor."""
    ops = conv_stack(0)
    x = noise(6, (1, 32, 8, 2))
    bs = bands(ops, x.shape, n_slices)
    from gale.slicing import heights

    h = heights(ops, x.shape)
    for t in range(len(ops) + 1):
        edges = [b[t] for b in bs]
        assert edges[0][0] == 0
        assert edges[-1][1] == h[t]
        for (_, hi), (lo, _) in zip(edges[:-1], edges[1:]):
            assert hi == lo


def test_overlap_that_is_enough_for_convolutions_is_not_enough_for_a_gate():
    """A squeeze-and-excite reduction reduces over the whole map, so the overlap
    that buys exactness for the local part of the stack leaves the sliced gate
    reading a per-slice statistic. Only a halo as wide as the tensor closes it,
    at which point the slicing has saved nothing. This is what G_A is for."""
    ops = gate_stack(0)
    local = [op for op in ops if not op.global_ctx]
    x = noise(7, (1, 32, 8, 2))
    o_in = receptive_field(local)
    exact_local, _ = sliced_forward(local, x, 4, overlap_profile(local, x.shape, o_in))
    assert float(jnp.max(jnp.abs(exact_local - exact_forward(local, x)))) < 1e-13

    ref = exact_forward(ops, x)
    y, _ = sliced_forward(ops, x, 4, overlap_profile(ops, x.shape, o_in))
    assert float(jnp.max(jnp.abs(y - ref))) > 1e-6
