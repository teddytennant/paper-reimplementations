"""Hand checks for MeanAbs, population Std, and column scales."""

import jax.numpy as jnp

from awt.importance import column_scales, importance_mean_abs, importance_std


def test_mean_abs_and_std_match_hand_values():
    # 3 samples, 2 channels.
    C = jnp.array(
        [
            [1.0, -2.0],
            [3.0, 0.0],
            [-1.0, 4.0],
        ]
    )
    eps = 1e-4

    got_abs = importance_mean_abs(C, eps)
    # channel 0: mean(1, 3, 1) = 5/3; channel 1: mean(2, 0, 4) = 2
    expected_abs = jnp.array([5.0 / 3.0, 2.0])
    assert got_abs.shape == (2,)
    assert jnp.allclose(got_abs, expected_abs, rtol=1e-5, atol=1e-5)
    # MeanAbs does not use eps.
    assert jnp.allclose(got_abs, importance_mean_abs(C, eps=0.25), rtol=1e-5, atol=1e-5)

    # Population variance, ddof=0.
    # channel 0 values 1, 3, -1, mean 1: squared devs 0, 4, 4, var = 8/3
    # channel 1 values -2, 0, 4, mean 2/3:
    #   devs -8/3, -2/3, 10/3; sum of squares 168/9; var = (168/9)/3 = 56/9
    var = jnp.array([8.0 / 3.0, 56.0 / 9.0])
    got_std = importance_std(C, eps)
    expected_std = jnp.sqrt(var + eps)
    assert got_std.shape == (2,)
    assert jnp.allclose(got_std, expected_std, rtol=1e-5, atol=1e-5)
    # eps sits inside the square root, not outside it.
    assert not jnp.allclose(got_std, jnp.sqrt(var) + eps, rtol=1e-5, atol=1e-5)
    # Not the sample variance (ddof=1): channel 0 would be 8/2 = 4, not 8/3.
    sample_var = jnp.array([4.0, 168.0 / 18.0])
    assert not jnp.allclose(got_std, jnp.sqrt(sample_var + eps), rtol=1e-4, atol=1e-4)


def test_column_scales_monotone_floor_and_hand_arithmetic():
    eps = 1e-6
    min_scale = 1e-3
    imp = jnp.array([1.0, 3.0])
    # mean = 2, so imp_eps = 2, s_min = 1e-3
    # alpha=1: (1/2, 3/2) = (0.5, 1.5)
    # alpha=2: (0.25, 2.25)
    s1 = column_scales(imp, alpha=1.0, eps=eps, min_scale=min_scale)
    s2 = column_scales(imp, alpha=2.0, eps=eps, min_scale=min_scale)
    assert jnp.allclose(s1, jnp.array([0.5, 1.5]), rtol=1e-5, atol=1e-5)
    assert jnp.allclose(s2, jnp.array([0.25, 2.25]), rtol=1e-5, atol=1e-5)
    assert s1[1] >= s1[0]
    assert s2[1] >= s2[0]

    # Floor binds on a tiny channel. mean of (0, 2) is 1, raw ratio is 0.
    tiny = jnp.array([0.0, 2.0])
    s_floor = column_scales(tiny, alpha=1.0, eps=eps, min_scale=min_scale)
    s_min = jnp.maximum(jnp.asarray(min_scale), jnp.asarray(eps))
    assert jnp.allclose(s_floor[0], s_min, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(s_floor[1], jnp.asarray(2.0), rtol=1e-5, atol=1e-5)
    assert s_floor[1] >= s_floor[0]

    # When eps exceeds min_scale, the floor is eps.
    s_eps_floor = column_scales(tiny, alpha=1.0, eps=1e-2, min_scale=1e-4)
    assert jnp.allclose(s_eps_floor[0], jnp.asarray(1e-2), rtol=1e-5, atol=1e-6)
    assert jnp.allclose(s_eps_floor[1], jnp.asarray(2.0), rtol=1e-5, atol=1e-5)
