"""The G_A branch and the alpha fusion: Section 3.3, Table 1, Appendix C."""

import jax.numpy as jnp
import pytest
from _helpers import conv_stack, gate_stack, mixed_input, noise

from gale import (
    ALPHA_DEFAULT,
    bilinear_down,
    exact_forward,
    ga_branch,
    gale_forward,
    mse,
    overlap_profile,
    sliced_forward,
)

GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def stack(name, seed):
    return conv_stack(seed) if name == "conv" else gate_stack(seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("name", ["conv", "gate"])
def test_ga_plus_le_beats_ga_alone(name, seed):
    """Table 1: G_A + L_E over G_A only. G_A runs on a bilinear downsample, so
    row-alternating detail is not in it at all; mixing the slices back in has to
    help on an input that carries such detail."""
    ops = stack(name, seed)
    x = mixed_input(50 + seed, h=48, w=12, c=2, high=1.5, low=1.0)
    ref = exact_forward(ops, x)
    halos = overlap_profile(ops, x.shape, 6)
    ga_only = mse(gale_forward(ops, x, 6, halos, 2, 0.0)[0], ref)
    both = mse(gale_forward(ops, x, 6, halos, 2, ALPHA_DEFAULT)[0], ref)
    assert both < ga_only


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["conv", "gate"])
def test_best_alpha_is_interior(name, seed):
    """Appendix C: the alpha curve peaks inside the interval. alpha = 1 is pure
    slicing with no global context, alpha = 0 leans entirely on a map that lost
    its high frequencies, and on an input carrying both kinds of structure the
    mix beats both ends.

    Where the optimum sits depends on how the content is split between the two,
    so the assertion is that it is interior, not that it is at 0.8. The 0.8 of
    Appendix C is an ImageNet number and there is no reason a synthetic input
    should land there.
    """
    ops = stack(name, seed)
    x = mixed_input(60 + seed, h=48, w=12, c=2)
    ref = exact_forward(ops, x)
    errs = [mse(gale_forward(ops, x, 6, None, 2, a)[0], ref) for a in GRID]
    best = min(range(len(GRID)), key=errs.__getitem__)
    assert 0 < best < len(GRID) - 1, dict(zip(GRID, errs))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["conv", "gate"])
def test_error_is_unimodal_in_alpha(name, seed):
    """One minimum, no second dip, so a calibrated alpha is well defined."""
    ops = stack(name, seed)
    x = mixed_input(60 + seed, h=48, w=12, c=2)
    ref = exact_forward(ops, x)
    errs = [mse(gale_forward(ops, x, 6, None, 2, a)[0], ref) for a in GRID]
    best = min(range(len(GRID)), key=errs.__getitem__)
    assert all(a > b for a, b in zip(errs[:best], errs[1 : best + 1])), errs
    assert all(a < b for a, b in zip(errs[best:-1], errs[best + 1 :])), errs


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_downsampling_keeps_a_global_statistic_that_slicing_destroys(seed):
    """Section 3.3, "G_A feature maps maintain global information".

    The squeeze-and-excite gate is driven by the mean of the whole map. Read off
    a band, that mean is wrong by a lot; read off a map downsampled 2x, it is
    close. This is the quantity the G_A branch exists to carry.
    """
    ops = gate_stack(seed)
    prefix = ops[:2]
    x = mixed_input(70 + seed, h=48, w=12, c=2)
    feat = exact_forward(prefix, x)
    whole = jnp.mean(feat, axis=(1, 2))
    ga = jnp.mean(exact_forward(prefix, bilinear_down(x, 2)), axis=(1, 2))
    step = feat.shape[1] // 8
    per_band = jnp.stack(
        [jnp.mean(feat[:, i * step : (i + 1) * step], axis=(1, 2)) for i in range(8)]
    )
    ga_err = float(jnp.max(jnp.abs(ga - whole)))
    band_err = float(jnp.max(jnp.abs(per_band - whole)))
    assert 4 * ga_err < band_err


def test_alpha_one_is_the_le_only_configuration():
    """The "-Le" rows of Appendix E: no G_A branch, no fusion, no extra cost."""
    ops = conv_stack(0)
    x = noise(80, (1, 32, 10, 2))
    halos = overlap_profile(ops, x.shape, 2)
    a, _ = gale_forward(ops, x, 4, halos, ga_factor=2, alpha=1.0)
    b, _ = sliced_forward(ops, x, 4, halos)
    assert float(jnp.max(jnp.abs(a - b))) == 0.0


def test_downsampling_is_where_the_ga_branch_saves_memory():
    ops = conv_stack(0)
    x = noise(81, (1, 32, 10, 2))
    _, tr = ga_branch(ops, x, 2, out_hw=(16, 5))
    _, full = ga_branch(ops, x, 1, out_hw=(16, 5))
    assert bilinear_down(x, 2).size * 4 == x.size
    assert tr.pair_peak * 3 < full.pair_peak
