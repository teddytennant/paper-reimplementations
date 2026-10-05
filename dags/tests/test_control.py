"""Independent numpy checks for the control equations."""

import numpy as np

from dags.control import (
    backward_warp,
    demodulate,
    flow_source,
    inject,
    lighting_recurrence,
    remodulate,
    temporal_guidance,
    temporal_lambda,
)


def test_demodulate_matches_numpy():
    rng = np.random.default_rng(0)
    image = rng.uniform(0.05, 2.0, size=(2, 5, 4, 3)).astype(np.float32)
    albedo = rng.uniform(0.05, 2.0, size=(2, 5, 4, 3)).astype(np.float32)
    eps = 1e-4
    # Reference written here, not by calling the library.
    ref = np.log(np.clip(image / albedo, eps, None))
    got = np.asarray(demodulate(image, albedo, eps=eps))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_demodulate_default_eps_clips_tiny_ratio():
    image = np.full((2, 2, 1), 1e-8, dtype=np.float32)
    albedo = np.ones((2, 2, 1), dtype=np.float32)
    got = np.asarray(demodulate(image, albedo))
    ref = np.log(np.full((2, 2, 1), 1e-4, dtype=np.float32))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_remodulate_formula_and_roundtrip():
    albedo = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    lighting = np.array([[[[0.0, np.log(3.0)]]]], dtype=np.float32)
    got = np.asarray(remodulate(albedo, lighting))
    np.testing.assert_allclose(got, [[[[1.0, 6.0]]]], rtol=1e-5, atol=1e-5)

    rng = np.random.default_rng(1)
    image = rng.uniform(0.2, 1.5, size=(6, 6, 3)).astype(np.float32)
    albedo = rng.uniform(0.2, 1.5, size=(6, 6, 3)).astype(np.float32)
    eps = 1e-4
    lighting = demodulate(image, albedo, eps=eps)
    recovered = np.asarray(remodulate(albedo, lighting))
    # Where the ratio was not clipped, the roundtrip recovers I.
    mask = (image / albedo) >= eps
    np.testing.assert_allclose(recovered[mask], image[mask], rtol=1e-5, atol=1e-5)
    # Clipped pixels come back as a * eps, not the original tiny ratio.
    tiny = np.full((2, 2, 1), 1e-8, dtype=np.float32)
    ones = np.ones((2, 2, 1), dtype=np.float32)
    clipped = np.asarray(remodulate(ones, demodulate(tiny, ones, eps=eps)))
    np.testing.assert_allclose(clipped, ones * eps, rtol=1e-5, atol=1e-7)


def test_warp_unique_pixel_one_pixel_shift():
    image = np.zeros((5, 6, 1), dtype=np.float32)
    image[2, 3, 0] = 7.0
    motion = np.zeros((5, 6, 2), dtype=np.float32)
    motion[..., 0] = 1.0  # dx = 1, sample x - 1, content moves right
    out = np.asarray(backward_warp(image, motion))
    assert out.shape == image.shape
    assert out[2, 4, 0] == np.float32(7.0)
    assert out[2, 3, 0] == np.float32(0.0)
    # Left column samples x = -1 and must be zero.
    assert np.all(out[:, 0, 0] == 0.0)
    # No other pixel holds the unique value.
    assert np.count_nonzero(out) == 1


def test_warp_dy_and_zero_motion_identity():
    image = np.zeros((4, 4, 2), dtype=np.float32)
    image[1, 2, :] = np.array([3.0, 4.0], dtype=np.float32)
    motion = np.zeros((4, 4, 2), dtype=np.float32)
    motion[..., 1] = 1.0  # dy = 1, content moves down
    out = np.asarray(backward_warp(image, motion))
    np.testing.assert_array_equal(out[2, 2], image[1, 2])
    assert np.all(out[0] == 0.0)

    rng = np.random.default_rng(3)
    dense = rng.normal(size=(5, 6, 2)).astype(np.float32)
    ident = np.asarray(backward_warp(dense, np.zeros((5, 6, 2), dtype=np.float32)))
    np.testing.assert_array_equal(ident, dense)


def test_warp_half_pixel_hand():
    image = np.zeros((2, 3, 1), dtype=np.float32)
    image[0, 1, 0] = 10.0
    image[1, 1, 0] = 2.0
    motion = np.zeros((2, 3, 2), dtype=np.float32)
    motion[..., 0] = 0.5
    out = np.asarray(backward_warp(image, motion))
    # output[0, 1] samples x=0.5: 0.5 * 0 + 0.5 * 10 = 5
    # output[0, 2] samples x=1.5: 0.5 * 10 + 0.5 * 0 = 5
    # output[1, 1] samples x=0.5: 0.5 * 0 + 0.5 * 2 = 1
    # output[0, 0] samples x=-0.5: outside contribution is zero
    np.testing.assert_allclose(out[0, 1, 0], 5.0, atol=1e-5)
    np.testing.assert_allclose(out[0, 2, 0], 5.0, atol=1e-5)
    np.testing.assert_allclose(out[1, 1, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(out[0, 0, 0], 0.0, atol=1e-5)


def test_lighting_recurrence_both_branches_against_numpy():
    rng = np.random.default_rng(2)
    height, width, channels = 4, 5, 2
    L_star = rng.normal(size=(height, width, channels)).astype(np.float32)
    g = rng.uniform(0.1, 0.9, size=(height, width, channels)).astype(np.float32)
    L_prev = rng.normal(size=(height, width, channels)).astype(np.float32)
    # Large delta so a wrong "always add delta" implementation cannot hide.
    delta = np.full((height, width, channels), 5.0, dtype=np.float32)
    motion = np.zeros((height, width, 2), dtype=np.float32)
    motion[..., 0] = 1.0
    valid = np.zeros((height, width), dtype=np.float32)
    valid[:, 2:] = 1.0

    # Numpy reference, including its own integer warp. Does not call the library.
    warped = np.zeros_like(L_prev)
    warped[:, 1:, :] = L_prev[:, :-1, :]
    fused = L_star + g * (warped - L_star) + delta
    expected = np.where(valid[..., None] > 0, fused, L_star)

    got = np.asarray(lighting_recurrence(L_star, g, L_prev, delta, motion, valid))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    # Invalid pixels are exactly the spatial estimate: no gate, no delta.
    np.testing.assert_array_equal(got[:, :2], L_star[:, :2])
    # Valid pixels actually take the fused branch.
    assert not np.allclose(got[:, 2:], L_star[:, 2:])

    # valid shaped (H, W, 1) broadcasts the same way.
    got_col = np.asarray(
        lighting_recurrence(L_star, g, L_prev, delta, motion, valid[..., None])
    )
    np.testing.assert_array_equal(got_col, got)


def test_temporal_lambda_endpoints_and_hand_value():
    # sigma = 0 gives lam_max.
    np.testing.assert_allclose(float(temporal_lambda(0.0, 0.7, 1.0)), 0.7, rtol=1e-6, atol=1e-6)
    # sigma = sigma_max and sigma > sigma_max both give 0.
    assert float(temporal_lambda(1.0, 0.7, 1.0)) == 0.0
    assert float(temporal_lambda(1.5, 0.7, 1.0)) == 0.0
    assert float(temporal_lambda(2.0, 0.4, 0.5)) == 0.0
    # Hand value: 0.4 * relu(1 - 0.25 / 1) = 0.3.
    np.testing.assert_allclose(float(temporal_lambda(0.25, 0.4, 1.0)), 0.3, atol=1e-6)


def test_temporal_guidance_hand_computation():
    x_hat = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
    x_prev = np.array([[[0.0], [2.0]], [[4.0], [8.0]]], dtype=np.float32)
    motion = np.zeros((2, 2, 2), dtype=np.float32)
    mask = np.array([[1.0, 0.0], [1.0, 0.5]], dtype=np.float32)
    sigma, lam_max, sigma_max = 0.25, 0.4, 1.0
    lam = 0.3  # hand value of lam(sigma)
    warped = x_prev  # zero motion
    expected = x_hat + lam * mask[..., None] * (warped - x_hat)
    got = np.asarray(temporal_guidance(x_hat, x_prev, motion, mask, sigma, lam_max, sigma_max))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    # Past sigma_max the step is a no-op.
    held = np.asarray(temporal_guidance(x_hat, x_prev, motion, mask, 3.0, lam_max, sigma_max))
    np.testing.assert_array_equal(held, x_hat)


def test_flow_source_returns_input_and_is_deterministic():
    relit = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    out1 = flow_source(relit)
    out2 = flow_source(relit)
    assert out1 is relit
    assert out2 is relit
    np.testing.assert_array_equal(np.asarray(out1), relit)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_inject_text_tokens_unchanged_and_local():
    rng = np.random.default_rng(4)
    n_tokens, dim = 6, 4
    h = rng.normal(size=(n_tokens, dim)).astype(np.float32)
    c_g = rng.normal(size=(n_tokens, dim)).astype(np.float32)
    c_a = rng.normal(size=(n_tokens, dim)).astype(np.float32)
    # First two tokens are text.
    mask = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    s_g, s_a = np.float32(0.5), np.float32(-0.25)
    got = np.asarray(inject(h, c_g, c_a, s_g, s_a, mask))
    expected = h + mask[:, None] * (s_g * c_g + s_a * c_a)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    # Text tokens are bit-identical to the input.
    np.testing.assert_array_equal(got[:2], h[:2])

    # (N, 1) mask broadcasts the same way.
    got_col = np.asarray(inject(h, c_g, c_a, s_g, s_a, mask[:, None]))
    np.testing.assert_array_equal(got_col, got)

    # Changing one image token's conditioning does not change any other token.
    c_g2 = c_g.copy()
    c_g2[3] = c_g2[3] + 1.0
    got2 = np.asarray(inject(h, c_g2, c_a, s_g, s_a, mask))
    np.testing.assert_array_equal(np.delete(got2, 3, axis=0), np.delete(got, 3, axis=0))
    assert not np.allclose(got2[3], got[3])

    # A text-token change in c_g is ignored.
    c_g3 = c_g.copy()
    c_g3[0] = c_g3[0] + 10.0
    got3 = np.asarray(inject(h, c_g3, c_a, s_g, s_a, mask))
    np.testing.assert_array_equal(got3, got)
