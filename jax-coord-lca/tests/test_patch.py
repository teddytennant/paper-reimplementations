import jax
import jax.numpy as jnp

from coord_lca.coords import fourier_embed, normalize_coords
from coord_lca.decoder import build_queries, fuse_query, init_decoder
from coord_lca.patch import extract_patches, init_patch_embed, patch_and_coord_features


def test_extract_patches_top_left_aligned():
    image = jnp.arange(8 * 8 * 3, dtype=jnp.float32).reshape(8, 8, 3)
    xy = jnp.array([[2.0, 1.0]])
    patches = extract_patches(image, xy, patch_size=3)
    assert patches.shape == (1, 3, 3, 3)
    assert jnp.allclose(patches[0], image[1:4, 2:5])


def test_extract_patches_zero_pad():
    image = jnp.ones((4, 4, 1), dtype=jnp.float32)
    xy = jnp.array([[3.0, 3.0]])
    patches = extract_patches(image, xy, patch_size=2)
    # top-left at (3, 3): one pixel inside, rest padded
    assert patches.shape == (1, 2, 2, 1)
    assert float(patches[0, 0, 0, 0]) == 1.0
    assert float(patches[0, 0, 1, 0]) == 0.0
    assert float(patches[0, 1, 0, 0]) == 0.0


def test_patch_coord_fusion_query_dim_matches_d_model():
    key = jax.random.PRNGKey(0)
    d_model = 32
    d_patch = 16
    n_freq = 6
    decoder = init_decoder(
        key,
        d_model=d_model,
        d_patch=d_patch,
        n_freq=n_freq,
        n_classes=3,
        patch_size=8,
        n_heads=4,
    )
    image = jax.random.uniform(jax.random.PRNGKey(1), (16, 16, 3))
    xy = jnp.array([[4.0, 4.0], [12.0, 8.0]])
    queries = build_queries(decoder, image, xy, patch_size=8, n_freq=n_freq)
    assert queries.shape == (2, d_model)

    f_p, f_c = patch_and_coord_features(
        decoder["patch"], image, xy, patch_size=8, n_freq=n_freq
    )
    assert f_p.shape == (2, d_patch)
    assert f_c.shape == (2, 4 * n_freq)
    fused = fuse_query(decoder["fuse"], f_p, f_c)
    assert fused.shape == (2, d_model)
    assert jnp.allclose(fused, queries)


def test_patch_embed_init_output_dim():
    key = jax.random.PRNGKey(2)
    params = init_patch_embed(key, patch_size=8, in_channels=3, d_patch=24)
    image = jnp.zeros((16, 16, 3), dtype=jnp.float32)
    xy = jnp.array([[1.0, 2.0]])
    f_p, f_c = patch_and_coord_features(params, image, xy, patch_size=8, n_freq=4)
    assert f_p.shape[-1] == 24
    assert f_c.shape[-1] == 16
    assert jnp.allclose(f_c, fourier_embed(normalize_coords(xy, 16, 16), n_freq=4))
