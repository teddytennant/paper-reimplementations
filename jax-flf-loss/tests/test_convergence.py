"""Does the frequency-first schedule actually get there sooner, at toy scale.

The paper reports up to 40% faster convergence in FID on ImageNet with JiT
models (Figure 4, Table 2). Nothing at this scale reproduces that number. What
is testable here is the direction of Section 4.2: start on the frequency
objective, switch to the pixel objective, and the high-frequency band is learned
sooner than it is under v-loss alone.

Score: relative error power in the 0.25 to 0.333 cycles/pixel annulus, from
Figure 6 (right), normalized by the target's power in the same annulus so 1.0
means "carries no information here". That annulus is the hardest part of this
data. It holds the least energy of the 1/f^2 background and none of the planted
detail modes, which both objectives pick up almost immediately.

Two runs of the same network on the same data with different seeds differ, so
every claim below is over 5 seeds and is asserted on the median and on the
per-seed win count, never on one run.

Measured, 5 seeds, 900 steps, switch at 450:
    v-loss  @600: 0.9906 to 1.0059, median 0.9970
    fv-loss @600: 0.9591 to 0.9727, median 0.9607
    v-loss  @900: 0.9727 to 0.9837, median 0.9787
    fv-loss @900: 0.9327 to 0.9396, median 0.9374
fv-loss wins on all 5 seeds at 900, and its 600-step score beats v-loss's
900-step score on all 5, which is the 1.5x fewer steps claim below.

Honest limits. The gap is real and it is small: 0.04 in a normalized error that
starts at 1.0, on a 0.26M-parameter MLP over 16x16 images. f-loss and v-loss
also differ by orders of magnitude in raw scale, and Adam's per-parameter
normalization does not fully absorb that across the schedule transient, so this
test cannot separate the spectral mechanism from an optimizer-scale effect. The
mechanism claim is tested directly, and deterministically, in
test_spectral_balance.py; this one only checks that the direction survives into
an actual training run.
"""

import numpy as np
import pytest

from flf.train import run_training

SEEDS = range(5)
STEPS = 900
SWITCH = 450
EARLY = 600


@pytest.fixture(scope="module")
def scores():
    v, fv = {}, {}
    for seed in SEEDS:
        v[seed], _ = run_training(seed, steps=STEPS, eval_at=(EARLY, STEPS))
        fv[seed], _ = run_training(
            seed, steps=STEPS, switch_step=SWITCH, eval_at=(EARLY, STEPS)
        )
    return v, fv


def _at(history, step):
    return np.array([history[seed][step] for seed in SEEDS])


def test_the_metric_starts_near_one_and_moves(scores):
    """Sanity: both objectives learn something in this band, neither diverges."""
    v, fv = scores
    for history in (v, fv):
        end = _at(history, STEPS)
        assert np.all(end < 1.0)
        assert np.all(end > 0.5)


def test_fv_loss_beats_v_loss_at_equal_step_count(scores):
    v, fv = scores
    v_end, fv_end = _at(v, STEPS), _at(fv, STEPS)
    assert np.median(fv_end) < np.median(v_end) - 0.02
    assert int(np.sum(fv_end < v_end)) >= 4


def test_fv_loss_reaches_v_loss_final_error_in_fewer_steps(scores):
    """fv-loss at 600 steps against v-loss at 900, a 1.5x budget cut."""
    v, fv = scores
    v_end, fv_early = _at(v, STEPS), _at(fv, EARLY)
    assert np.median(fv_early) < np.median(v_end)
    assert int(np.sum(fv_early < v_end)) >= 4


def test_the_gap_is_already_open_before_the_switch_completes(scores):
    """Section 4.2: the frequency stage is what buys the early lead."""
    v, fv = scores
    assert np.median(_at(fv, EARLY)) < np.median(_at(v, EARLY)) - 0.02
