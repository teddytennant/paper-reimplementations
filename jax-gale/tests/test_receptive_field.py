"""Equation 1 and the PPBI overlap it defines."""

import math

import jax
import jax.numpy as jnp
import pytest
from _helpers import conv, linear_stack, noise

from gale import calibration_step, exact_forward, ppbi_overlap, receptive_field
from gale.layers import GlobalGate, Relu


def influence_width(ops, x, out_row):
    """How many input rows actually reach one output row, read off the gradient."""
    g = jax.grad(lambda z: jnp.sum(exact_forward(ops, z)[:, out_row]))(x)
    rows = jnp.max(jnp.abs(g), axis=(0, 2, 3)) > 1e-12
    idx = jnp.nonzero(rows)[0]
    return int(idx[-1] - idx[0]) + 1


@pytest.mark.parametrize("seed", [0, 1])
def test_equation_1_matches_measured_receptive_field(seed):
    """R_0(L) is the number of input rows an interior output element depends on."""
    ops = linear_stack(seed)
    x = noise(seed, (1, 40, 9, 2))
    y = exact_forward(ops, x)
    mid = y.shape[1] // 2
    assert influence_width(ops, x, mid) == receptive_field(ops)


def test_receptive_field_recurses_through_the_stack():
    """R_0(L) built in one shot equals R_0(t) composed with R_t(L)."""
    ops = linear_stack(0)
    for t in range(len(ops) + 1):
        head = receptive_field(ops, 0, t)
        tail = receptive_field(ops, t, len(ops))
        stride_head = math.prod(op.row_s for op in ops[:t])
        assert receptive_field(ops) == head + (tail - 1) * stride_head


def test_ppbi_overlap_is_infinite_for_a_global_operator():
    """Section 3: PPBI is not expensive on squeeze-and-excite, it is inapplicable."""
    gate = GlobalGate(gain=jnp.ones((3,)), bias=jnp.zeros((3,)))
    ops = linear_stack(0) + [gate]
    assert ppbi_overlap(linear_stack(0)) < math.inf
    assert ppbi_overlap(ops) == math.inf


def test_calibration_increment_follows_algorithm_1():
    """R_i(i+1) across a convolution, s_{i+1} across anything else."""
    ops = [conv(jax.random.PRNGKey(0), 2, 2, 5, 2), Relu()]
    assert calibration_step(ops, 0) == 5
    assert calibration_step(ops, 1) == 1
