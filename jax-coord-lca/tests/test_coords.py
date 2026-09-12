import jax.numpy as jnp

from coord_lca.coords import (
    FOURIER_OUT_MULTIPLIER,
    dense_query_coords,
    fourier_embed,
    fourier_frequencies,
    image_to_encoder_grid,
    normalize_coords,
)


def test_fourier_output_dim():
    n_freq = 6
    xy = jnp.array([[0.25, 0.75], [0.0, 1.0]])
    emb = fourier_embed(xy, n_freq=n_freq)
    assert emb.shape == (2, FOURIER_OUT_MULTIPLIER * n_freq)
    assert emb.shape[-1] == 4 * n_freq


def test_fourier_known_frequencies():
    n_freq = 4
    freqs = fourier_frequencies(n_freq)
    expected = jnp.array([jnp.pi, 2 * jnp.pi, 4 * jnp.pi, 8 * jnp.pi])
    assert jnp.allclose(freqs, expected)
    xy = jnp.array([0.25, 0.5])
    emb = fourier_embed(xy, n_freq=n_freq)
    x, y = 0.25, 0.5
    want = jnp.concatenate(
        [
            jnp.sin(freqs * x),
            jnp.cos(freqs * x),
            jnp.sin(freqs * y),
            jnp.cos(freqs * y),
        ]
    )
    assert jnp.allclose(emb, want, atol=1e-6)


def test_fourier_trig_range_and_distinct():
    grid = jnp.stack(jnp.meshgrid(jnp.linspace(0, 1, 9), jnp.linspace(0, 1, 9), indexing="ij"), axis=-1)
    emb = fourier_embed(grid, n_freq=8)
    assert float(emb.min()) >= -1.0 - 1e-6
    assert float(emb.max()) <= 1.0 + 1e-6
    a = fourier_embed(jnp.array([0.0, 0.0]), n_freq=6)
    b = fourier_embed(jnp.array([1.0, 0.0]), n_freq=6)
    assert not jnp.allclose(a, b)


def test_normalize_coords():
    xy = jnp.array([[0.0, 0.0], [16.0, 8.0], [32.0, 16.0]])
    norm = normalize_coords(xy, height=16, width=32)
    assert jnp.allclose(norm, jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))


def test_encoder_cell_center_maps_to_half_index():
    # 32x32 image, downsample=2, patch=4 => encoder 4x4.
    height, width = 32, 32
    downsample, p_e = 2, 4
    he, we = 4, 4
    # Token (i, j) occupies image rectangle of size (s_H * p_e) x (s_W * p_e)
    # = 8 x 8. Center of cell (i=1, j=2) is x=2*8+4=20, y=1*8+4=12.
    i, j = 1, 2
    cell = downsample * p_e
    xy = jnp.array([j * cell + cell / 2.0, i * cell + cell / 2.0])
    mapped = image_to_encoder_grid(
        xy,
        height=height,
        width=width,
        downsample=downsample,
        encoder_patch_size=p_e,
    )
    assert jnp.allclose(mapped, jnp.array([j + 0.5, i + 0.5]), atol=1e-5)
    mapped2 = image_to_encoder_grid(
        xy, height=height, width=width, encoder_height=he, encoder_width=we
    )
    assert jnp.allclose(mapped, mapped2)


def test_image_corners_map_to_grid_corners():
    height, width = 32, 48
    he, we = 4, 6
    corners = jnp.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(height)],
            [float(width), float(height)],
        ]
    )
    mapped = image_to_encoder_grid(
        corners, height=height, width=width, encoder_height=he, encoder_width=we
    )
    want = jnp.array(
        [
            [0.0, 0.0],
            [float(we), 0.0],
            [0.0, float(he)],
            [float(we), float(he)],
        ]
    )
    assert jnp.allclose(mapped, want, atol=1e-5)


def test_eq7_downsample_form():
    # eq. 7: (x_down, y_down) = (x / (s_W p_e), y / (s_H p_e))
    xy = jnp.array([12.0, 20.0])
    mapped = image_to_encoder_grid(
        xy, height=32, width=32, downsample=2, encoder_patch_size=4
    )
    assert jnp.allclose(mapped, jnp.array([12.0 / 8.0, 20.0 / 8.0]))


def test_dense_query_coords_centers():
    coords = dense_query_coords(2, 3, centers=True)
    assert coords.shape == (2, 3, 2)
    assert jnp.allclose(coords[0, 0], jnp.array([0.5, 0.5]))
    assert jnp.allclose(coords[1, 2], jnp.array([2.5, 1.5]))
