"""The full pipeline on the paper's motivating example, the negated Branin
function of Figure 1.

Two things get checked. The candidates have to leave the offline data behind,
and the UGTL constructor has to produce the trajectory geometry Figure 1(b) and
Figure 2 claim for it, measured against the incoherent constructor with every
other stage held fixed.
"""

import dataclasses

import jax
import numpy as np
import pytest

from ugtl import UGTLConfig, branin_task, run_ugtl
from ugtl.diagnostics import cumulative_regret, smoothness

CONFIG = UGTLConfig(n_traj=200, ensemble_steps=600, diffusion_steps=2500, n_gen=192, Q=8)


@pytest.fixture(scope="module")
def runs():
    out = []
    for seed in (0, 1):
        k_task, k_run = jax.random.split(jax.random.PRNGKey(seed))
        task = branin_task(k_task, n_samples=4000, keep_fraction=0.6)
        out.append((task, {c: run_ugtl(k_run, task, CONFIG, c) for c in ("ugtl", "random")}))
    return out


def test_candidates_beat_the_best_design_in_the_offline_dataset(runs):
    for task, res in runs:
        scores = task.evaluate(res["ugtl"].candidates)
        assert len(scores) == CONFIG.Q
        assert scores.max() > task.best_offline
        assert np.median(scores) > task.best_offline


def test_candidates_stay_inside_the_design_box(runs):
    for task, res in runs:
        c = res["ugtl"].candidates
        assert (c >= task.lo - 1e-6).all() and (c <= task.hi + 1e-6).all()


def test_the_constructor_produces_much_smoother_paths(runs):
    """S(tau) of Section 5.3.1, the smoothness column of Figure 2.

    This is the separation that does not depend on the setting. Restricting
    transitions to NN_K of the score-feasible set bounds every step by the local
    neighbourhood radius, so the second difference stays small, while a random
    walk over the dataset crosses the box on every step.
    """
    for task, res in runs:
        s_u = smoothness(res["ugtl"].trajectories.designs, task.lo, task.hi).mean()
        s_r = smoothness(res["random"].trajectories.designs, task.lo, task.hi).mean()
        assert s_u < 0.05 * s_r


def test_the_constructor_climbs_further_at_this_scale(runs):
    """Lower R(tau) and higher terminal scores than the ablation.

    Both depend on the horizon relative to the density of the offline data. The
    K nearest feasible neighbours sit closer together as the dataset grows, so a
    fixed H buys less climb, while the best of H uniform draws only improves.
    These numbers are for the 2400 offline designs and H = 32 that ``CONFIG``
    fixes; ``scripts/smoke.py`` runs H = 48 on 15000 designs, where the regret
    ordering goes the other way.
    """
    for _, res in runs:
        u, r = res["ugtl"].trajectories, res["random"].trajectories
        # run_ugtl hands the constructor scores already scaled to [0, 1], which
        # is the s_k that R(tau) is defined against.
        assert cumulative_regret(u.scores).mean() < 0.99 * cumulative_regret(r.scores).mean()
        assert u.scores[:, -1].mean() > r.scores[:, -1].mean()
        assert u.improvement.mean() > r.improvement.mean()


def test_the_target_score_is_lambda_times_the_observed_maximum():
    """y_tar = lambda_tar max y, above everything in the data."""
    k_task, k_run = jax.random.split(jax.random.PRNGKey(4))
    task = branin_task(k_task, n_samples=1500, keep_fraction=0.6)
    cfg = dataclasses.replace(CONFIG, n_traj=60, diffusion_steps=400, n_gen=64, Q=4)
    res = run_ugtl(k_run, task, cfg)
    assert res.y_tar == pytest.approx(cfg.lambda_tar)  # scores are scaled to [0, 1]
    assert res.y_tar > 1.0


def test_an_unknown_constructor_is_rejected():
    task = branin_task(jax.random.PRNGKey(0), n_samples=400, keep_fraction=0.6)
    with pytest.raises(ValueError):
        run_ugtl(jax.random.PRNGKey(0), task, CONFIG, "nope")
