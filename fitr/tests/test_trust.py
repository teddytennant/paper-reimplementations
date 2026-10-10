import jax.numpy as jnp

from fitr import box, sides, step_length


def test_low_sensitivity_gets_longer_side():
    side = sides(jnp.array([1.0, 100.0]), 0.8)
    assert float(side[0]) > float(side[1])


def test_sides_have_target_volume():
    kappa = 0.8
    side = sides(jnp.array([1.0, 100.0]), kappa)
    assert abs(float(jnp.prod(side)) - kappa ** 2) < 1e-6


def test_box_clips_to_unit_cube():
    lo, hi = box(jnp.array([0.05, 0.95]), jnp.array([0.4, 0.4]))
    assert jnp.all(lo >= 0.0)
    assert jnp.all(hi <= 1.0)
    assert float(lo[0]) == 0.0
    assert float(hi[1]) == 1.0
    assert abs(float(hi[0]) - 0.25) < 1e-6
    assert abs(float(lo[1]) - 0.75) < 1e-6


def test_ten_improvements_double_and_cap():
    kappa, succ, fail, restarted = 0.8, 0, 0, False
    for _ in range(9):
        kappa, succ, fail, restarted = step_length(kappa, True, succ, fail, d=2)
    assert kappa == 0.8
    assert succ == 9
    kappa, succ, fail, restarted = step_length(kappa, True, succ, fail, d=2)
    assert kappa == 1.6
    assert succ == 0
    assert fail == 0
    assert restarted is False


def test_failures_halve_then_restart():
    kappa, succ, fail, restarted = 0.8, 0, 0, False
    for _ in range(4):
        kappa, succ, fail, restarted = step_length(kappa, False, succ, fail, d=2)
    assert kappa == 0.4
    assert fail == 0
    assert succ == 0
    assert restarted is False

    # 0.8 / 2**7 = 0.00625 < 0.5**7, so the 7th halving restarts.
    for _ in range(6 * 4 - 1):
        kappa, succ, fail, restarted = step_length(kappa, False, succ, fail, d=2)
        assert restarted is False
    kappa, succ, fail, restarted = step_length(kappa, False, succ, fail, d=2)
    assert restarted is True
    assert kappa == 0.8
    assert succ == 0
    assert fail == 0
