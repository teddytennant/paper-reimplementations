"""Equation tests for MARS-CLIP (arXiv:2609.08283)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marsclip.attention import (
    combine_biases,
    key_key_logits,
    linear,
    refined_attention_block,
    structure_aware_attention,
)
from marsclip.color import (
    COLOR_SIGMA,
    color_affinity,
    color_bias,
    patch_lab_colors,
    srgb_to_lab,
)
from marsclip.constants import ALPHA, ENCODER_SIZE, S_CROP
from marsclip.fusion import (
    encode_multiresolution,
    fuse_features,
    grid_crops,
    grid_shape,
    pad_to_multiple,
    reassemble_local_features,
    resize_chw,
    upsample_features,
)
from marsclip.scores import cosine_scores, predict_mask

jax.config.update("jax_enable_x64", False)


def _np(x) -> np.ndarray:
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Fusion Eq. 1
# ---------------------------------------------------------------------------


def test_fusion_eq1_matches_alpha_up_plus_local():
    rng = np.random.default_rng(0)
    f_global = jnp.asarray(rng.normal(size=(4, 5, 8)).astype(np.float32))
    f_local = jnp.asarray(rng.normal(size=(12, 15, 8)).astype(np.float32))
    alpha = 0.8
    refined = fuse_features(f_global, f_local, alpha)
    up_global = upsample_features(f_global, (12, 15))
    expected = alpha * up_global + (1.0 - alpha) * f_local
    np.testing.assert_allclose(_np(refined), _np(expected), rtol=1e-5, atol=1e-5)
    assert refined.shape == f_local.shape


def test_fusion_alpha_extremes():
    f_global = jnp.ones((2, 2, 3), dtype=jnp.float32)
    f_local = jnp.full((4, 4, 3), 7.0, dtype=jnp.float32)
    only_local = fuse_features(f_global, f_local, alpha=0.0)
    np.testing.assert_allclose(_np(only_local), 7.0, atol=1e-6)
    only_global = fuse_features(f_global, f_local, alpha=1.0)
    up = upsample_features(f_global, (4, 4))
    np.testing.assert_allclose(_np(only_global), _np(up), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Grid / reflective pad
# ---------------------------------------------------------------------------


def test_reflective_pad_makes_dims_multiples_of_s_crop():
    image = jnp.arange(3 * 100 * 150, dtype=jnp.float32).reshape(3, 100, 150)
    padded = pad_to_multiple(image, S_CROP)
    _, h, w = padded.shape
    assert h % S_CROP == 0
    assert w % S_CROP == 0
    n_rows, n_cols = grid_shape((h, w), S_CROP)
    assert n_rows * S_CROP == h
    assert n_cols * S_CROP == w
    # Original content is preserved in the unpadded region.
    np.testing.assert_array_equal(_np(padded[:, :100, :150]), _np(image))
    # Already-aligned images are unchanged.
    aligned = jnp.zeros((3, 224, 336), dtype=jnp.float32)
    np.testing.assert_array_equal(
        _np(pad_to_multiple(aligned, S_CROP)), _np(aligned)
    )


def test_grid_crops_n_m_times_s_crop_equals_padded_size():
    image = jnp.arange(3 * 80 * 90, dtype=jnp.float32).reshape(3, 80, 90)
    padded = pad_to_multiple(image, s_crop=32)
    crops = grid_crops(padded, s_crop=32)
    n_rows, n_cols, channels, ch, cw = crops.shape
    assert channels == 3
    assert ch == 32 and cw == 32
    assert n_rows * 32 == padded.shape[1]
    assert n_cols * 32 == padded.shape[2]
    # First crop equals the top-left tile of the padded image.
    np.testing.assert_array_equal(_np(crops[0, 0]), _np(padded[:, :32, :32]))


def test_resize_and_reassemble_with_callable_encoder():
    """Helper grids/pads/resizes; encoder is a callable on KxK crops."""
    s_crop = 4
    k = 8
    image = jnp.arange(3 * 6 * 5, dtype=jnp.float32).reshape(3, 6, 5) / 100.0

    def encoder(crop):
        # crop: (3, K, K) -> (2, 2, 3) mean-pooled 2x2 of the three channels.
        assert crop.shape == (3, k, k)
        pooled = crop.reshape(3, 2, k // 2, 2, k // 2).mean(axis=(2, 4))
        return jnp.moveaxis(pooled, 0, -1)

    refined = encode_multiresolution(
        image, encoder, s_crop=s_crop, encoder_size=k, alpha=ALPHA
    )
    padded = pad_to_multiple(image, s_crop)
    n_rows, n_cols = grid_shape(padded.shape[1:], s_crop)
    assert refined.shape == (n_rows * 2, n_cols * 2, 3)


# ---------------------------------------------------------------------------
# 3. KK^T attention with B = 0
# ---------------------------------------------------------------------------


def test_kk_attention_equals_softmax_when_bias_zero():
    rng = np.random.default_rng(1)
    keys = jnp.asarray(rng.normal(size=(6, 4)).astype(np.float32))
    attn = structure_aware_attention(keys)
    d = keys.shape[-1]
    logits = keys @ keys.T / jnp.sqrt(jnp.asarray(d, dtype=keys.dtype))
    expected = jax.nn.softmax(logits, axis=-1)
    np.testing.assert_allclose(_np(attn), _np(expected), rtol=1e-5, atol=1e-5)
    row_sums = _np(attn).sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)


def test_key_key_logits_scale():
    keys = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    logits = key_key_logits(keys)
    expected = (keys @ keys.T) / jnp.sqrt(jnp.float32(2.0))
    np.testing.assert_allclose(_np(logits), _np(expected), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Bias add: A = softmax(KK^T/sqrt(d) + B_int + B_color)
# ---------------------------------------------------------------------------


def test_bias_add_numerically():
    rng = np.random.default_rng(2)
    keys = jnp.asarray(rng.normal(size=(5, 8)).astype(np.float32))
    b_int = jnp.asarray(rng.normal(size=(5, 5)).astype(np.float32))
    b_color = jnp.asarray(rng.normal(size=(5, 5)).astype(np.float32) * 0.1)
    attn = structure_aware_attention(keys, b_int=b_int, b_color=b_color)
    d = keys.shape[-1]
    logits = keys @ keys.T / jnp.sqrt(jnp.asarray(d, dtype=keys.dtype))
    expected = jax.nn.softmax(logits + b_int + b_color, axis=-1)
    np.testing.assert_allclose(_np(attn), _np(expected), rtol=1e-5, atol=1e-5)
    # Explicit B_total path matches the split-bias path.
    attn_total = structure_aware_attention(keys, b_total=b_int + b_color)
    np.testing.assert_allclose(_np(attn), _np(attn_total), atol=1e-6)
    np.testing.assert_allclose(
        _np(combine_biases(b_int, b_color)), _np(b_int + b_color), atol=0
    )


# ---------------------------------------------------------------------------
# 5. Color Gaussian Eq. 3
# ---------------------------------------------------------------------------


def test_color_gaussian_identical_colors_affinity_one():
    colors = jnp.asarray(
        [
            [50.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [20.0, 10.0, -5.0],
        ],
        dtype=jnp.float32,
    )
    affinity = color_affinity(colors, sigma=COLOR_SIGMA)
    np.testing.assert_allclose(_np(affinity[0, 1]), 1.0, atol=1e-6)
    np.testing.assert_allclose(_np(affinity[1, 0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.diag(_np(affinity)), 1.0, atol=1e-6)
    # Off-diagonal for distinct colors is the Gaussian of squared L2.
    delta = colors[0] - colors[2]
    dist2 = float(jnp.sum(delta * delta))
    expected = np.exp(-dist2 / (2.0 * COLOR_SIGMA**2))
    np.testing.assert_allclose(_np(affinity[0, 2]), expected, atol=1e-6)


def test_color_bias_cls_row_col_zero():
    colors = jnp.asarray(
        [
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [[12.0, 22.0, 32.0], [13.0, 23.0, 33.0]],
        ],
        dtype=jnp.float32,
    )
    bias = color_bias(colors, sigma=COLOR_SIGMA, has_cls_token=True)
    n_spatial = 4
    assert bias.shape == (n_spatial + 1, n_spatial + 1)
    np.testing.assert_array_equal(_np(bias[0, :]), 0.0)
    np.testing.assert_array_equal(_np(bias[:, 0]), 0.0)
    spatial = color_affinity(colors.reshape(-1, 3), sigma=COLOR_SIGMA)
    np.testing.assert_allclose(_np(bias[1:, 1:]), _np(spatial), atol=1e-6)


def test_patch_lab_colors_tiny_grid():
    # 4x4 image, patch_size=2 -> 2x2 patch colors.
    image = jnp.zeros((3, 4, 4), dtype=jnp.float32)
    image = image.at[0, :2, :2].set(1.0)  # red block
    colors = patch_lab_colors(image, patch_size=2)
    assert colors.shape == (2, 2, 3)
    # Top-left patch is saturated red; others are black.
    red_lab = srgb_to_lab(jnp.asarray([1.0, 0.0, 0.0]))
    black_lab = srgb_to_lab(jnp.asarray([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(_np(colors[0, 0]), _np(red_lab), atol=1e-4)
    np.testing.assert_allclose(_np(colors[0, 1]), _np(black_lab), atol=1e-4)


# ---------------------------------------------------------------------------
# 6. sRGB to CIELAB
# ---------------------------------------------------------------------------


# Reference Lab (D65, 2°) for IEC 61966-2-1 sRGB, rounded to 0.01.
# Sources: CIE formulae; cross-checked against Bruce Lindbloom / EasyRGB.
_SRGB_LAB_REFERENCE = [
    # white
    ([1.0, 1.0, 1.0], (100.0, 0.0, 0.0)),
    # black
    ([0.0, 0.0, 0.0], (0.0, 0.0, 0.0)),
    # saturated red
    ([1.0, 0.0, 0.0], (53.24, 80.09, 67.20)),
    # saturated green
    ([0.0, 1.0, 0.0], (87.73, -86.18, 83.18)),
    # saturated blue
    ([0.0, 0.0, 1.0], (32.30, 79.19, -107.86)),
    # mid grey
    ([0.5, 0.5, 0.5], (53.39, 0.0, 0.0)),
]


@pytest.mark.parametrize("srgb, lab_ref", _SRGB_LAB_REFERENCE)
def test_srgb_to_lab_known_values(srgb, lab_ref):
    lab = srgb_to_lab(jnp.asarray(srgb, dtype=jnp.float32))
    np.testing.assert_allclose(_np(lab), np.asarray(lab_ref), atol=1.0)


def test_srgb_to_lab_chw_layout():
    image = jnp.stack(
        [
            jnp.ones((2, 2)),
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
        ]
    )
    lab = srgb_to_lab(image)
    assert lab.shape == (3, 2, 2)
    red = srgb_to_lab(jnp.asarray([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(_np(lab[:, 0, 0]), _np(red), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Cosine score Eq. 4 + argmax
# ---------------------------------------------------------------------------


def test_cosine_score_eq4_and_argmax():
    # Two pixels, two classes. Pixel (0,0) aligned with class 0, (0,1) with class 1.
    f_refined = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    f_text = jnp.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    scores = cosine_scores(f_refined, f_text)
    # Cosine of parallel vectors is 1, orthogonal is 0.
    np.testing.assert_allclose(_np(scores[0, 0]), np.array([1.0, 0.0]), atol=1e-5)
    np.testing.assert_allclose(_np(scores[0, 1]), np.array([0.0, 1.0]), atol=1e-5)
    labels = predict_mask(scores, target_hw=(1, 2))
    np.testing.assert_array_equal(_np(labels), np.array([[0, 1]]))


def test_cosine_scale_invariance_and_upsample_argmax():
    rng = np.random.default_rng(3)
    feats = jnp.asarray(rng.normal(size=(3, 3, 6)).astype(np.float32))
    text = jnp.asarray(rng.normal(size=(4, 6)).astype(np.float32))
    s1 = cosine_scores(feats, text)
    s2 = cosine_scores(3.5 * feats, 0.25 * text)
    np.testing.assert_allclose(_np(s1), _np(s2), atol=1e-5)
    labels = predict_mask(s1, target_hw=(6, 6))
    assert labels.shape == (6, 6)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 4


# ---------------------------------------------------------------------------
# 8. Final block has no residual
# ---------------------------------------------------------------------------


def test_final_block_is_linear_av_not_residual():
    rng = np.random.default_rng(4)
    keys = jnp.asarray(rng.normal(size=(4, 5)).astype(np.float32))
    values = jnp.asarray(rng.normal(size=(4, 5)).astype(np.float32))
    weight = jnp.asarray(rng.normal(size=(5, 7)).astype(np.float32))
    bias = jnp.asarray(rng.normal(size=(7,)).astype(np.float32))
    x = jnp.asarray(rng.normal(size=(4, 7)).astype(np.float32))
    out = refined_attention_block(keys, values, weight, bias)
    attn = structure_aware_attention(keys)
    expected = linear(attn @ values, weight, bias)
    np.testing.assert_allclose(_np(out), _np(expected), rtol=1e-5, atol=1e-5)
    residual = x + expected
    assert not np.allclose(_np(out), _np(residual), atol=1e-3)
    assert out.shape == (4, 7)


# ---------------------------------------------------------------------------
# 9. Permutation / scale sanity
# ---------------------------------------------------------------------------


def test_attention_row_permutation_equivariance():
    rng = np.random.default_rng(5)
    keys = jnp.asarray(rng.normal(size=(5, 6)).astype(np.float32))
    values = jnp.asarray(rng.normal(size=(5, 6)).astype(np.float32))
    perm = jnp.asarray([2, 0, 4, 1, 3])
    attn = structure_aware_attention(keys)
    attn_p = structure_aware_attention(keys[perm])
    # Permuting tokens permutes both axes of A.
    np.testing.assert_allclose(
        _np(attn_p), _np(attn[perm][:, perm]), rtol=1e-5, atol=1e-5
    )
    agg = attn @ values
    agg_p = attn_p @ values[perm]
    np.testing.assert_allclose(_np(agg_p), _np(agg[perm]), rtol=1e-5, atol=1e-5)


def test_reassemble_roundtrip_grid():
    rng = np.random.default_rng(6)
    local = jnp.asarray(rng.normal(size=(2, 3, 4, 5, 7)).astype(np.float32))
    assembled = reassemble_local_features(local)
    assert assembled.shape == (8, 15, 7)
    # Top-left tile is crop (0, 0).
    np.testing.assert_allclose(
        _np(assembled[:4, :5]), _np(local[0, 0]), atol=0
    )


def test_resize_chw_identity_when_same_size():
    image = jnp.arange(3 * 8 * 8, dtype=jnp.float32).reshape(3, 8, 8)
    out = resize_chw(image, 8)
    np.testing.assert_allclose(_np(out), _np(image), atol=1e-5)
