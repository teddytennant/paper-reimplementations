import jax.numpy as jnp
import numpy as np
import pytest

from cadmp.masking import (
    GRAY,
    apply_mask,
    attention_mask,
    mask_image,
    token_grid_shape,
    upsample_attention,
)


def test_token_grid_shape():
    assert token_grid_shape(576) == (24, 24)
    assert token_grid_shape(256) == (16, 16)
    with pytest.raises(ValueError, match="not a square grid"):
        token_grid_shape(577)


def test_nearest_upsampling_puts_one_hot_token_on_exactly_its_patch():
    """The token-to-pixel correspondence, isolated from any thresholding."""
    grid, patch = 4, 2
    for token in range(grid * grid):
        attn = np.zeros(grid * grid, dtype=np.float64)
        attn[token] = 1.0
        up = np.asarray(
            upsample_attention(jnp.asarray(attn), (grid * patch, grid * patch),
                               method="nearest")
        )
        row, col = divmod(token, grid)
        expected = np.zeros((grid * patch, grid * patch))
        expected[row * patch : (row + 1) * patch, col * patch : (col + 1) * patch] = 1.0
        np.testing.assert_array_equal(up, expected)


def test_a_single_hot_token_masks_exactly_its_patch_and_nothing_else():
    """One token above threshold, so the mask is that token's 2x2 pixel patch.

    tau = 100/16 selects the top 1/16 of pixels, and 4 of the 64 pixels are the
    hot token's patch, so the arithmetic is exact rather than rounded.
    """
    grid, patch = 4, 2
    rng = np.random.default_rng(0)
    attn = rng.uniform(0.001, 0.02, size=grid * grid)
    attn[9] = 1.0

    up = upsample_attention(jnp.asarray(attn), (8, 8), method="nearest")
    mask = np.asarray(attention_mask(up, tau=100.0 / 16.0))

    row, col = divmod(9, grid)
    expected = np.zeros((8, 8), dtype=bool)
    expected[row * patch : (row + 1) * patch, col * patch : (col + 1) * patch] = True
    np.testing.assert_array_equal(mask, expected)
    assert mask.sum() == 4


def test_tau_30_masks_exactly_thirty_percent():
    """10x10 tokens upsampled to 20x20 pixels: 400 pixels, 120 masked."""
    rng = np.random.default_rng(1)
    attn = rng.permutation(np.arange(100).astype(np.float64) + 1.0)

    up = upsample_attention(jnp.asarray(attn), (20, 20), method="nearest")
    mask = np.asarray(attention_mask(up, tau=30.0))

    assert mask.sum() == 120 == int(0.30 * 400)


def test_masked_positions_are_exactly_the_top_thirty_percent_by_attention():
    rng = np.random.default_rng(2)
    attn = rng.permutation(np.arange(100).astype(np.float64) + 1.0)

    up = upsample_attention(jnp.asarray(attn), (20, 20), method="nearest")
    mask = np.asarray(attention_mask(up, tau=30.0))

    top_tokens = set(np.argsort(attn)[-30:].tolist())
    expected = np.zeros((20, 20), dtype=bool)
    for token in top_tokens:
        row, col = divmod(token, 10)
        expected[row * 2 : row * 2 + 2, col * 2 : col * 2 + 2] = True

    np.testing.assert_array_equal(mask, expected)


def test_the_threshold_is_the_seventieth_percentile_not_the_thirtieth():
    """Masking the top of the attention map, not the bottom."""
    attn = jnp.asarray(np.arange(100, dtype=np.float64))
    mask = np.asarray(attention_mask(attn.reshape(10, 10), tau=30.0))
    assert mask.reshape(-1)[-1]
    assert not mask.reshape(-1)[0]


def test_masked_pixels_are_gray_and_the_rest_are_untouched():
    rng = np.random.default_rng(3)
    image = jnp.asarray(rng.uniform(0.0, 255.0, size=(20, 20, 3)))
    mask = jnp.asarray(rng.random((20, 20)) < 0.3)

    out = np.asarray(apply_mask(image, mask))
    m = np.asarray(mask)

    assert np.all(out[m] == GRAY)
    assert GRAY == 128.0
    np.testing.assert_array_equal(out[~m], np.asarray(image)[~m])


def test_apply_mask_handles_a_single_channel_image():
    image = jnp.asarray(np.arange(16.0).reshape(4, 4))
    mask = jnp.asarray(np.eye(4, dtype=bool))
    out = np.asarray(apply_mask(image, mask))
    np.testing.assert_array_equal(np.diag(out), np.full(4, GRAY))
    assert out[0, 1] == 1.0


def test_mask_image_end_to_end():
    rng = np.random.default_rng(4)
    image = jnp.asarray(rng.uniform(0.0, 255.0, size=(20, 20, 3)))
    attn = jnp.asarray(rng.permutation(np.arange(100).astype(np.float64) + 1.0))

    out = np.asarray(mask_image(image, attn, tau=30.0, method="nearest"))
    grayed = np.all(out == GRAY, axis=-1)

    assert grayed.sum() == 120
    np.testing.assert_array_equal(
        grayed, np.asarray(attention_mask(
            upsample_attention(attn, (20, 20), method="nearest"), tau=30.0))
    )


def test_bilinear_upsampling_stays_inside_the_token_range():
    rng = np.random.default_rng(5)
    attn = jnp.asarray(rng.random(576))
    up = np.asarray(upsample_attention(attn, (336, 336), method="bilinear"))
    assert up.shape == (336, 336)
    assert up.min() >= float(attn.min()) - 1e-9
    assert up.max() <= float(attn.max()) + 1e-9


def test_bilinear_and_nearest_disagree_so_the_choice_is_not_cosmetic():
    rng = np.random.default_rng(6)
    attn = jnp.asarray(rng.random(256))
    a = np.asarray(upsample_attention(attn, (64, 64), method="bilinear"))
    b = np.asarray(upsample_attention(attn, (64, 64), method="nearest"))
    assert not np.allclose(a, b)


def test_unknown_upsampling_method_is_rejected():
    with pytest.raises(ValueError, match="unknown upsampling method"):
        upsample_attention(jnp.ones(16), (8, 8), method="bicubic-ish")
