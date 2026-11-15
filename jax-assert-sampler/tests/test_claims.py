"""The empirical claims of Sections 3.1 and 4.2, on the 2D toy.

Every number here comes from a fixed seed. Each deployment realization is
evaluated with all samplers sharing that realization's eps_W and sampler key, the
protocol of Section 4.1, and the reported quantity is the mean over realizations.
"""

import jax
import jax.numpy as jnp
import pytest

from assert_sampler import schedules, toy

STEPS = 50
LEVEL = 0.03  # equivalent weight-noise standard deviation
MAPPINGS = 12


@pytest.fixture(scope="module")
def setup():
    return toy.setup(n_samples=1000)


@pytest.fixture(scope="module")
def named():
    return toy.default_schedules(STEPS)


@pytest.fixture(scope="module")
def fixed_noise_sweep(setup, named):
    return toy.sweep(setup, STEPS, LEVEL, named, num_mappings=MAPPINGS)


def test_clean_inference_is_close_to_the_target(setup, named):
    """Without hardware noise the learned denoiser already samples the mixture, so a
    degradation below cannot be blamed on the network."""
    out = toy.run(setup, STEPS, named["ddim"], jax.random.key(11))
    assert toy.error(setup, out) < 0.02


def test_assert_beats_deterministic_ddim_under_fixed_spatial_noise(fixed_noise_sweep):
    """The central claim of Section 4.2, at matched step count and matched parameters:
    only the eta schedule and the random draws differ between the two runs."""
    ddim_mean, ddim_each = fixed_noise_sweep["ddim"]
    assert_mean, assert_each = fixed_noise_sweep["assert"]
    assert ddim_mean > 0.5, "the noise level has to actually hurt for this to mean anything"
    assert assert_mean < 0.6 * ddim_mean
    wins = sum(a < d for a, d in zip(assert_each, ddim_each))
    assert wins >= 9


def test_early_stochastic_schedules_beat_late_ones(fixed_noise_sweep):
    """Figure 8's ordering. Reverse step keeps the same total stochasticity but spends
    it on final refinement, so it recovers only part of the gap."""
    m = {k: v[0] for k, v in fixed_noise_sweep.items()}
    assert m["assert"] < m["linear"] < m["reverse_step"] < m["ddim"]
    assert m["step"] < m["linear"]
    # Cosine and step decay are within noise of each other on this toy; both are
    # clearly ahead of reverse step.
    assert max(m["assert"], m["step"]) < 0.7 * m["reverse_step"]


def test_the_early_stage_is_the_vulnerable_one(setup, named):
    """Figure 4. Switch the hardware mapping on over a 20% window at the high-noise end
    or a 20% window at the refinement end, same Delta theta and same magnitude, and
    compare. This is the observation ASSERT is built on."""
    w = STEPS // 5
    early = jnp.concatenate([jnp.ones(w), jnp.zeros(STEPS - w)])
    late = jnp.concatenate([jnp.zeros(STEPS - w), jnp.ones(w)])
    det = {"ddim": named["ddim"]}
    e = toy.sweep(setup, STEPS, LEVEL, det, num_mappings=8, active=early)["ddim"]
    l = toy.sweep(setup, STEPS, LEVEL, det, num_mappings=8, active=late)["ddim"]
    assert e[0] > 5.0 * l[0]
    # and not because of one bad realization.
    assert min(e[1]) > max(l[1])


def test_persistence_is_what_makes_the_noise_expensive(setup, named, fixed_noise_sweep):
    """The control. Redrawing eps_W every step keeps the noise magnitude identical and
    destroys only its persistence. Deterministic DDIM stops being hurt much, and the
    absolute error ASSERT removes collapses with it.

    What does not shrink is the *ratio*: see the README note. On this toy, ASSERT
    still divides the resampled-noise error by a similar factor, so part of its gain
    is the generic self-correction any stochastic sampler gives against a wrong score,
    not decorrelation. The decorrelation part is isolated in
    ``tests/test_analysis.py`` instead, where the cross terms can be measured directly.
    """
    pair = {"ddim": named["ddim"], "assert": named["assert"]}
    free = toy.sweep(setup, STEPS, LEVEL, pair, num_mappings=MAPPINGS,
                     resample_per_step=True)
    assert fixed_noise_sweep["ddim"][0] > 3.0 * free["ddim"][0]
    gain_fixed = fixed_noise_sweep["ddim"][0] - fixed_noise_sweep["assert"][0]
    gain_free = free["ddim"][0] - free["assert"][0]
    assert gain_fixed > 2.0 * gain_free


def test_the_penalty_for_persistence_widens_with_step_count(setup):
    """Table 1's signature. Errors from one reused mapping stay correlated, so more
    denoising steps keep adding to the same directional error and deterministic DDIM
    gets worse. Redraw eps_W every step and the per-step errors partly cancel instead,
    so the gap between the two regimes grows with the step count."""
    err = {}
    for resample in (False, True):
        for s in (10, 100):
            etas = {"ddim": schedules.constant_schedule(s, 0.0)}
            err[(resample, s)] = toy.sweep(setup, s, LEVEL, etas, num_mappings=5,
                                           resample_per_step=resample)["ddim"][0]
    short = err[(False, 10)] / err[(True, 10)]
    long = err[(False, 100)] / err[(True, 100)]
    assert short > 1.5
    assert long > 2.0 * short
    assert err[(False, 100)] > 3.0 * err[(False, 10)]
