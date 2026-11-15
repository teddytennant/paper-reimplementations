"""Stand-in encoders, token align, per-layer scales, and the stabilizer step."""

import jax
import jax.numpy as jnp
import numpy as np

from dags.modules import (
    AppearanceEncoder,
    GeometryEncoder,
    StabilizerStep,
    init_layer_scales,
    inject_layers,
    token_align,
)


def test_geometry_stems_are_separate_arrays():
    params = GeometryEncoder.init(jax.random.PRNGKey(0), d_out=4)
    depth_k = params["depth_stem"]["kernel"]
    normal_k = params["normal_stem"]["kernel"]
    assert depth_k.shape == (3, 3, 1, 8)
    assert normal_k.shape == (3, 3, 3, 8)
    assert depth_k is not normal_k
    # Different shapes already mean different arrays. Values of a same-rank
    # slice also differ, so the stems were not copied from one draw.
    assert not np.array_equal(np.asarray(depth_k)[..., 0, :], np.asarray(normal_k)[..., 0, :])


def test_geometry_output_shape_and_both_stems_used():
    params = GeometryEncoder.init(jax.random.PRNGKey(1), d_out=6)
    depth = jnp.zeros((2, 7, 5, 1))
    normals = jnp.zeros((2, 7, 5, 3))
    base = GeometryEncoder.apply(params, depth, normals)
    assert base.shape == (2, 7, 5, 6)
    from_depth = GeometryEncoder.apply(params, jnp.ones_like(depth), normals)
    from_norm = GeometryEncoder.apply(params, depth, jnp.ones_like(normals))
    assert not np.allclose(from_depth, base)
    assert not np.allclose(from_norm, base)


def test_appearance_encoder_concat_and_shape():
    params = AppearanceEncoder.init(jax.random.PRNGKey(2), d_out=5)
    assert params["kernel"].shape == (3, 3, 9, 5)
    assert params["bias"].shape == (5,)
    albedo = jnp.ones((8, 8, 3))
    raw = jnp.ones((8, 8, 3))
    relit = jnp.zeros((8, 8, 3))
    out = AppearanceEncoder.apply(params, albedo, raw, relit)
    assert out.shape == (8, 8, 5)
    out_r = AppearanceEncoder.apply(params, albedo, raw, jnp.ones((8, 8, 3)))
    assert not np.allclose(out, out_r)


def test_token_align_shape_and_block_mean():
    feat = jnp.arange(4 * 4 * 2, dtype=jnp.float32).reshape(4, 4, 2)
    tokens = token_align(feat, 2)
    assert tokens.shape == (4, 2)
    # Top-left 2x2 block is token 0. Mean is written out, not via the library.
    block = np.asarray(feat)[:2, :2]
    expected = block.mean(axis=(0, 1))
    np.testing.assert_allclose(np.asarray(tokens[0]), expected, rtol=1e-6, atol=1e-6)
    # Bottom-right block is token 3 (row-major).
    block_br = np.asarray(feat)[2:, 2:]
    np.testing.assert_allclose(np.asarray(tokens[3]), block_br.mean(axis=(0, 1)), rtol=1e-6, atol=1e-6)

    batched = jnp.stack([feat, feat + 1.0], axis=0)
    batched_tokens = token_align(batched, 2)
    assert batched_tokens.shape == (2, 4, 2)


def test_layer_scales_start_at_zero_and_are_a_noop():
    scales = init_layer_scales(3)
    assert scales["s_g"].shape == (3,)
    assert scales["s_a"].shape == (3,)
    assert np.all(np.asarray(scales["s_g"]) == 0.0)
    assert np.all(np.asarray(scales["s_a"]) == 0.0)
    h = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    c_g = jnp.ones((4, 3))
    c_a = jnp.ones((4, 3)) * 2.0
    mask = jnp.array([0.0, 1.0, 1.0, 0.0])
    outs = inject_layers(h, c_g, c_a, scales, mask)
    assert isinstance(outs, list) and len(outs) == 3
    for out in outs:
        np.testing.assert_array_equal(np.asarray(out), np.asarray(h))


def test_layer_scales_are_independent():
    scales = init_layer_scales(2)
    h = jnp.zeros((5, 3))
    c_g = jnp.ones((5, 3))
    c_a = jnp.ones((5, 3)) * 0.25
    mask = jnp.ones((5,))
    base = inject_layers(h, c_g, c_a, scales, mask)
    scales_sg = {"s_g": scales["s_g"].at[0].set(1.5), "s_a": scales["s_a"]}
    outs = inject_layers(h, c_g, c_a, scales_sg, mask)
    # Changing s_g[0] does not change layer 1.
    np.testing.assert_array_equal(np.asarray(outs[1]), np.asarray(base[1]))
    assert not np.allclose(outs[0], base[0])
    # And changing s_a[1] does not change layer 0.
    scales_sa = {"s_g": scales["s_g"], "s_a": scales["s_a"].at[1].set(-2.0)}
    outs_a = inject_layers(h, c_g, c_a, scales_sa, mask)
    np.testing.assert_array_equal(np.asarray(outs_a[0]), np.asarray(base[0]))
    assert not np.allclose(outs_a[1], base[1])


def test_inject_layers_accepts_a_list_of_states():
    scales = {"s_g": jnp.array([1.0, 2.0]), "s_a": jnp.array([0.5, -1.0])}
    h0 = jnp.zeros((3, 2))
    h1 = jnp.ones((3, 2))
    c_g = jnp.ones((3, 2))
    c_a = jnp.full((3, 2), 4.0)
    mask = jnp.array([1.0, 0.0, 1.0])
    outs = inject_layers([h0, h1], c_g, c_a, scales, mask)
    assert len(outs) == 2
    # Hand residual per layer, masked. Text-like token 1 stays put.
    for index, (state, s_g, s_a) in enumerate([(h0, 1.0, 0.5), (h1, 2.0, -1.0)]):
        residual = s_g * np.asarray(c_g) + s_a * np.asarray(c_a)
        expected = np.asarray(state) + np.asarray(mask)[:, None] * residual
        np.testing.assert_allclose(np.asarray(outs[index]), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(outs[index])[1], np.asarray(state)[1])


def test_stabilizer_init_and_invalid_ignores_history():
    key = jax.random.PRNGKey(3)
    params = StabilizerStep.init(key, channels=3, std=0.02)
    expected_kernel = jax.random.normal(key, (3, 3, 3, 9)) * 0.02
    np.testing.assert_array_equal(np.asarray(params["kernel"]), np.asarray(expected_kernel))
    assert np.all(np.asarray(params["bias"]) == 0.0)

    frame = jnp.ones((8, 8, 3))
    motion = jnp.zeros((8, 8, 2))
    valid = jnp.zeros((8, 8))
    out1 = StabilizerStep.apply(params, frame, motion, valid, jnp.zeros((8, 8, 3)))
    out2 = StabilizerStep.apply(params, frame, motion, valid, jnp.ones((8, 8, 3)) * 4.0)
    assert out1.shape == (8, 8, 3)
    # Invalid everywhere: output is L_star, so history does not enter.
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    valid_ones = jnp.ones((8, 8))
    moved = jnp.zeros((8, 8, 2)).at[..., 0].set(1.0)
    hist = jnp.arange(8 * 8 * 3, dtype=jnp.float32).reshape(8, 8, 3)
    with_hist = StabilizerStep.apply(params, frame, moved, valid_ones, hist)
    without = StabilizerStep.apply(params, frame, moved, valid_ones, jnp.zeros_like(hist))
    assert with_hist.shape == frame.shape
    assert not np.allclose(with_hist, without)
