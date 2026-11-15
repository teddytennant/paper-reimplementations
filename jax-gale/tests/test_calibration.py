"""Section 3.1 and Algorithm 1: what the calibration pass promises."""

import pytest
from _helpers import conv_stack, gate_stack, noise, pool_stack

from gale import (
    calibrate,
    exact_forward,
    mse,
    overlap_profile,
    ppbi_overlap,
    sliced_forward,
)

EPSILONS = [1e-1, 1e-2, 1e-3, 1e-5, 1e-9]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_error_grows_as_overlap_shrinks(seed):
    """Reconstruction error is non-increasing in the overlap, and a slice with no
    overlap at all is strictly worse than one with the full receptive field."""
    ops = conv_stack(seed)
    x = noise(10 + seed, (1, 40, 10, 2))
    ref = exact_forward(ops, x)
    errs = []
    for o_in in range(0, 9):
        y, _ = sliced_forward(ops, x, 4, overlap_profile(ops, x.shape, o_in))
        errs.append(mse(y, ref))
    assert all(a >= b - 1e-15 for a, b in zip(errs[:-1], errs[1:])), errs
    assert errs[0] > errs[-1]
    assert errs[-1] == 0.0


@pytest.mark.parametrize("stack", ["conv", "pool"])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_calibration_delivers_its_error_target(stack, epsilon):
    """After calibrating at epsilon, the measured MSE of a fresh sliced pass with
    the returned overlaps is below epsilon."""
    ops = conv_stack(0) if stack == "conv" else pool_stack(0)
    x = noise(11, (1, 40, 10, 2))
    ref = exact_forward(ops, x)
    cal = calibrate(ops, x, epsilon, n_slices=4)
    assert cal.met
    y, _ = sliced_forward(ops, x, cal.n_slices, cal.halos)
    assert mse(y, ref) <= epsilon
    assert mse(y, ref) == pytest.approx(cal.mse, abs=1e-15)


def test_chosen_overlap_is_monotone_in_epsilon():
    """A looser target never costs more overlap than a tighter one."""
    ops = conv_stack(0)
    x = noise(11, (1, 40, 10, 2))
    totals = [calibrate(ops, x, e, n_slices=4).total_overlap for e in sorted(EPSILONS)]
    assert all(a >= b for a, b in zip(totals[:-1], totals[1:])), totals
    assert totals[0] > totals[-1]


def test_calibrated_overlap_stays_under_the_ppbi_overlap():
    """O_GaLe <= O_ppbi, Section 3.1, even at an epsilon small enough to force
    bit-exact reconstruction."""
    ops = conv_stack(0)
    x = noise(12, (1, 40, 10, 2))
    cal = calibrate(ops, x, 1e-12, n_slices=4)
    assert cal.mse == 0.0
    assert cal.halos[0] <= ppbi_overlap(ops)
    assert sum(cal.halos) < len(cal.halos) * ppbi_overlap(ops)


def test_overlap_profile_decays_with_depth():
    """The per-layer overlaps shrink as the maps shrink; PPBI would carry R_0(L)
    rows at every layer."""
    ops = conv_stack(0)
    x = noise(12, (1, 40, 10, 2))
    cal = calibrate(ops, x, 1e-12, n_slices=4)
    assert list(cal.halos) == sorted(cal.halos, reverse=True)
    assert cal.halos[-1] < cal.halos[0]


def test_memory_limit_forces_more_subdivisions():
    """Algorithm 1's outer loop: when the overlap will not fit, N goes up."""
    ops = conv_stack(0)
    x = noise(13, (1, 40, 10, 2))
    loose = calibrate(ops, x, 1e-12, n_slices=2, budget_elems=10**7, max_slices=8)
    tight = calibrate(ops, x, 1e-12, n_slices=2, budget_elems=950, max_slices=8)
    assert loose.n_slices == 2
    assert tight.n_slices > loose.n_slices
    assert tight.met


def test_a_global_operator_only_calibrates_by_giving_up_on_slicing():
    """With no memory limit the pass can only reach zero error on a stack with a
    global reduction by widening the halo until every slice holds the whole
    tensor, which saves nothing. Put a real budget on it and it reports failure
    rather than pretending. Recovering the global statistic is the G_A branch's
    job, not the overlap's."""
    ops = gate_stack(0)
    x = noise(14, (1, 32, 8, 2))
    free = calibrate(ops, x, 1e-12, n_slices=4)
    band = x.shape[1] / 4
    assert 2 * free.halos[0] + band >= x.shape[1]

    capped = calibrate(ops, x, 1e-12, n_slices=4, budget_elems=1400, max_slices=8)
    assert not capped.met
    assert capped.mse > 0
