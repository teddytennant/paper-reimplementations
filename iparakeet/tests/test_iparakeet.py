"""CPU checks for Phi, eq. 10/11 fusion, minimax Swish, and range remedies."""

import jax
import jax.numpy as jnp

from iparakeet.attention import (
    activation_scales,
    float_attention,
    fuse_scores_dyadic,
    fuse_scores_reference,
    integer_attention,
    rel_shift,
)
from iparakeet.range import dequantize, percentile_scale, quantize
from iparakeet.swish import SWISH_A, SWISH_C, hard_swish, swish, swish_hat, tanh_hat


def test_phi_scales_exactly():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 9))
    scale = jnp.asarray(2.5)
    got = rel_shift(scale * x)
    expect = scale * rel_shift(x)
    assert jnp.allclose(got, expect, atol=0, rtol=0)


def test_phi_known_pattern():
    # qlen=3, klen=2. A zero is inserted into the kept columns.
    narrow = jnp.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    assert jnp.array_equal(
        rel_shift(narrow),
        jnp.array(
            [
                [3, 0],
                [5, 6],
                [7, 8],
            ]
        ),
    )
    # qlen=3, klen=3. Standard pad / reshape / drop-row / take-klen.
    wide = jnp.arange(1, 16).reshape(3, 5)
    assert jnp.array_equal(
        rel_shift(wide),
        jnp.array(
            [
                [3, 4, 5],
                [7, 8, 9],
                [11, 12, 13],
            ]
        ),
    )


def test_dyadic_matches_floor_within_one():
    Sc, Sp, Ss, dk = 0.05, 0.03, 0.02, 4
    key = jax.random.PRNGKey(1)
    for i in range(5):
        k1, k2, key = jax.random.split(key, 3)
        qc = jax.random.randint(k1, (4, 4), -40, 41, dtype=jnp.int8)
        qp = jax.random.randint(k2, (4, 7), -40, 41, dtype=jnp.int8)
        ref = fuse_scores_reference(qc, qp, Sc, Sp, Ss, dk)
        dy = fuse_scores_dyadic(qc, qp, Sc, Sp, Ss, dk, n=16)
        gap = jnp.max(jnp.abs(ref - dy.astype(ref.dtype)))
        assert int(gap) <= 1, (i, float(gap))


def test_phi_commutes_with_position_scale():
    key = jax.random.PRNGKey(2)
    qp = jax.random.randint(key, (4, 7), -40, 41, dtype=jnp.int8)
    Sp = jnp.asarray(0.03)
    left = rel_shift(qp).astype(jnp.float32) * Sp
    right = Sp * rel_shift(qp).astype(jnp.float32)
    assert jnp.allclose(left, right, atol=0, rtol=0)


def test_swish_grid_and_hard_swish():
    assert SWISH_A == -0.1240
    assert SWISH_C == 2.4632
    xs = jnp.linspace(-8.0, 8.0, 20001)
    err_hat = float(jnp.max(jnp.abs(swish_hat(xs) - swish(xs))))
    err_hard = float(jnp.max(jnp.abs(hard_swish(xs) - swish(xs))))
    assert 0.02 <= err_hat <= 0.045, err_hat
    assert err_hard > 0.10, err_hard
    assert err_hat < err_hard


def test_tanh_hat_odd_and_swish_hat_zero():
    u = jnp.linspace(-4.0, 4.0, 1001)
    assert jnp.allclose(tanh_hat(-u), -tanh_hat(u), atol=0, rtol=0)
    assert float(swish_hat(jnp.asarray(0.0))) == 0.0


def test_percentile_scale_beats_minmax_on_inliers():
    key = jax.random.PRNGKey(3)
    inliers = jax.random.normal(key, (1000,))
    outlier = 1000.0 * jnp.max(jnp.abs(inliers))
    x = jnp.concatenate([inliers, outlier[None]])
    alpha = percentile_scale(x, 99.9)
    assert jnp.allclose(alpha, percentile_scale(x))
    max_abs = jnp.max(jnp.abs(x))
    assert float(alpha) < 0.1 * float(max_abs)

    def inlier_mse(clip):
        codes, step = quantize(x, 8, alpha=clip)
        rec = dequantize(codes, step, dtype=x.dtype)
        return jnp.mean((rec[:-1] - x[:-1]) ** 2)

    mse_pct = inlier_mse(alpha)
    mse_mm = inlier_mse(max_abs)
    assert float(mse_pct) < float(mse_mm)


def test_int16_reconstruction_finer_than_int8():
    x = jnp.linspace(-1.0, 1.0, 2048)
    c8, s8 = quantize(x, 8)
    c16, s16 = quantize(x, 16)
    mse8 = jnp.mean((dequantize(c8, s8, dtype=x.dtype) - x) ** 2)
    mse16 = jnp.mean((dequantize(c16, s16, dtype=x.dtype) - x) ** 2)
    assert float(mse16) < float(mse8)


def test_integer_attention_close_to_float():
    key = jax.random.PRNGKey(4)
    keys = jax.random.split(key, 6)
    length, dk, dv = 4, 8, 4
    Q = jax.random.normal(keys[0], (length, dk)) * 0.5
    K = jax.random.normal(keys[1], (length, dk)) * 0.5
    V = jax.random.normal(keys[2], (length, dv))
    P = jax.random.normal(keys[3], (2 * length - 1, dk)) * 0.5
    u = jax.random.normal(keys[4], (dk,)) * 0.1
    v = jax.random.normal(keys[5], (dk,)) * 0.1
    scales = activation_scales(Q, K, V, P, u, v)
    got = integer_attention(Q, K, V, P, u, v, scales)
    ref = float_attention(Q, K, V, P, u, v)
    err = jnp.max(jnp.abs(got - ref))
    assert bool(jnp.isfinite(err))
    assert float(err) < 0.5, float(err)
