"""Peak activation memory, measured from the arrays the implementation actually
builds rather than from the formula in the paper.

Both paths report the same live-set model: an op holds its input and its output
at the same time, and a sliced pass also holds the full input and output buffers
of the block it is tiling. Every number below comes from ``.size`` on a
real array.
"""

import jax.numpy as jnp
import pytest
from _helpers import noise, wide_stack

from gale import (
    exact_forward,
    exact_forward_trace,
    full_map_elems,
    gale_attention,
    overlap_profile,
    plan_adaptive,
    run_plan,
    sliced_forward,
    tensor_shapes,
)
from gale.attention import attention_map


def wide_case():
    return wide_stack(0), noise(20, (1, 64, 16, 1))


def test_sliced_peak_is_lower_than_the_whole_map_peak_at_equal_output():
    """The headline claim, on a stack whose early activations dominate: same
    output to the last bit, strictly smaller peak."""
    ops, x = wide_case()
    ref, exact_tr = exact_forward_trace(ops, x)
    halos = overlap_profile(ops, x.shape, 4)
    y, tr = sliced_forward(ops, x, 8, halos)
    assert float(jnp.max(jnp.abs(y - ref))) < 1e-13
    assert tr.peak < exact_tr.peak
    assert max(tr.arrays) < max(exact_tr.arrays)


@pytest.mark.parametrize("halo", [0, 4])
def test_peak_falls_as_slices_are_added(halo):
    ops, x = wide_case()
    halos = overlap_profile(ops, x.shape, halo)
    peaks = [sliced_forward(ops, x, n, halos)[1].peak for n in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(peaks[:-1], peaks[1:])), peaks


def test_adaptive_slicing_uses_fewer_slices_as_the_maps_shrink():
    """Section 3.2: the patch count drops with the resolution instead of being
    fixed across the whole network."""
    ops, x = wide_case()
    plan = plan_adaptive(ops, x.shape, 2048, starts=(0, 3, 5))
    assert list(plan.n_slices) == sorted(plan.n_slices, reverse=True)
    assert plan.n_slices[0] > plan.n_slices[-1]
    y, tr = run_plan(ops, x, plan)
    assert y.shape == exact_forward(ops, x).shape
    assert tr.peak < exact_forward_trace(ops, x)[1].peak


def test_a_block_plan_respects_its_budget():
    """Every window a block materializes fits the budget, on a stack whose
    largest activation is eight times that budget. The tensors between blocks are
    a separate cost, which is why the split points sit where the maps are small."""
    ops, x = wide_case()
    budget = 2048
    plan = plan_adaptive(ops, x.shape, budget, starts=(0, 3, 5))
    _, tr = run_plan(ops, x, plan)
    assert max(tr.arrays) <= budget
    assert max(s[1] * s[2] * s[3] for s in tensor_shapes(ops, x.shape)) >= 8 * budget


@pytest.mark.parametrize("b", [2, 4])
def test_attention_peak_map_is_the_subset_map(b):
    """Section 4.1.2: b + 1 maps of size (N/b) x (N/b), one at a time, instead of
    one N x N map."""
    n, d = 16, 4
    q, k, v = (noise(30 + i, (n, d)).reshape(n, d) for i in range(3))
    _, peak = gale_attention(q, k, v, b=b, alpha=0.8, return_peak=True)
    assert peak == (n // b) ** 2
    assert attention_map(q, k).size == full_map_elems(n)
    assert peak * (b + 1) < full_map_elems(n)
