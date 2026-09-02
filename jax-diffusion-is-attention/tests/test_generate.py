"""Sec. V, Algorithms 1 and 2 on CPU-scale synthetic latents."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffusion_is_attention import (
    IDENTITY_CODEC,
    Codec,
    SimilarityTree,
    exact_topk,
    generate,
    spatial_attention,
)

SHAPE = (3, 7, 7)  # the MNIST latent shape of Table II


def latents(seed=0, n=128, shape=SHAPE):
    return jax.random.normal(jax.random.key(seed), (n,) + shape)


def test_alg2_weights_sum_to_one_at_every_location():
    """Line 8 of Alg. 2 is a softmax over the neighbour index, not over pixels."""
    s = latents(1, n=5)
    q = latents(2, n=1)[0]
    w = spatial_attention(q, s, sigma=0.5)
    assert w.shape == (5, SHAPE[1], SHAPE[2])
    assert jnp.all(w >= 0.0)
    assert jnp.allclose(jnp.sum(w, axis=0), 1.0, atol=1e-12)


def test_alg2_prefers_the_neighbour_that_matches_at_that_location():
    """Two neighbours, each an exact copy of the query on half the image."""
    q = latents(3, n=1)[0]
    a = q.at[:, :, 3:].set(q[:, :, 3:] + 5.0)  # matches on the left half
    b = q.at[:, :, :3].set(q[:, :, :3] + 5.0)  # matches on the right half
    w = spatial_attention(q, jnp.stack([a, b]), sigma=0.5)
    assert jnp.all(w[0, :, :3] > 0.99)
    assert jnp.all(w[1, :, 3:] > 0.99)


def test_alg2_is_uniform_when_all_neighbours_are_equidistant():
    q = jnp.zeros(SHAPE)
    s = jnp.stack([jnp.full(SHAPE, 1.0), jnp.full(SHAPE, -1.0)])
    w = spatial_attention(q, s, sigma=0.3)
    assert jnp.allclose(w, 0.5, atol=1e-12)


def test_alg2_sigma_controls_sharpness():
    q = latents(4, n=1)[0]
    s = latents(5, n=4)
    sharp = spatial_attention(q, s, sigma=0.05)
    flat = spatial_attention(q, s, sigma=50.0)
    assert float(jnp.max(sharp)) > float(jnp.max(flat))
    assert jnp.allclose(flat, 1.0 / 4.0, atol=2e-3)


def test_alg1_line8_sets_the_latent_norm_to_r():
    """Line 3 computes r as the mean latent radius, line 8 rescales z to it."""
    s = latents(6)
    out = generate(s, s[0], k=4, sigma=0.5, eta=0.0)
    r = float(jnp.mean(jnp.linalg.norm(s.reshape(s.shape[0], -1), axis=-1)))
    assert float(out.radius) == pytest.approx(r, rel=1e-12)
    assert float(jnp.linalg.norm(out.rescaled_latent)) == pytest.approx(r, rel=1e-7)


def test_alg1_k1_no_noise_identity_codec_returns_the_rescaled_neighbour():
    s = latents(7)
    q = s[11] + 0.01 * jax.random.normal(jax.random.key(12), SHAPE)
    out = generate(s, q, k=1, sigma=0.5, eta=0.0, codec=IDENTITY_CODEC)
    assert list(out.neighbor_indices) == [11]
    assert jnp.allclose(out.weights, 1.0, atol=1e-12)
    expect = out.radius * s[11] / jnp.linalg.norm(s[11])
    assert jnp.allclose(out.image, expect, rtol=1e-7, atol=1e-9)
    assert jnp.allclose(out.image, out.latent, atol=0.0)


def test_alg1_neighbours_are_the_true_topk():
    s = latents(8)
    q = latents(9, n=1)[0]
    out = generate(s, q, k=5, sigma=0.5, eta=0.0)
    want = exact_topk(s.reshape(s.shape[0], -1), q.reshape(-1), 5).indices
    assert list(out.neighbor_indices) == list(want)


def test_alg1_line9_noise_scales_with_eta():
    s = latents(10)
    q = s[3]
    key = jax.random.key(21)
    clean = generate(s, q, k=3, sigma=0.5, eta=0.0)
    noisy = generate(s, q, k=3, sigma=0.5, eta=0.05, key=key)
    assert jnp.allclose(clean.rescaled_latent, noisy.rescaled_latent, atol=1e-12)
    drift = jnp.linalg.norm(noisy.latent - clean.latent)
    expect = 0.05 * jnp.sqrt(jnp.asarray(np.prod(SHAPE), dtype=drift.dtype))
    assert float(drift) == pytest.approx(float(expect), rel=0.15)
    with pytest.raises(ValueError):
        generate(s, q, k=3, sigma=0.5, eta=0.05)


def test_alg1_output_lies_in_the_cone_of_its_neighbours():
    """Lines 6 and 7 are a per-location convex combination, so no new extremes."""
    s = jnp.abs(latents(13))
    q = s[5]
    out = generate(s, q, k=4, sigma=0.5, eta=0.0)
    picked = s[out.neighbor_indices]
    scale = out.radius / jnp.linalg.norm(jnp.sum(out.weights[:, None] * picked, axis=0))
    lo = jnp.min(picked, axis=0) * scale
    hi = jnp.max(picked, axis=0) * scale
    assert jnp.all(out.rescaled_latent >= lo - 1e-9)
    assert jnp.all(out.rescaled_latent <= hi + 1e-9)


def test_alg1_codec_is_a_plug():
    """Quantize, decode and the three AE passes of lines 10 to 14 all get called."""
    calls = {"quantize": 0, "decode": 0, "refine": 0}

    def counted(name, fn):
        def wrapped(x):
            calls[name] += 1
            return fn(x)

        return wrapped

    codec = Codec(
        quantize=counted("quantize", lambda z: jnp.round(z * 4) / 4),
        decode=counted("decode", lambda z: jnp.sum(z, axis=0)),
        refine=counted("refine", lambda x: 0.5 * x),
    )
    s = latents(14)
    out = generate(s, s[2], k=3, sigma=0.5, eta=0.0, codec=codec)
    assert calls == {"quantize": 1, "decode": 1, "refine": 3}
    assert out.image.shape == SHAPE[1:]
    assert jnp.allclose(out.image, 0.125 * jnp.sum(jnp.round(out.latent * 4) / 4, axis=0))


def test_alg1_through_the_tree_matches_the_brute_force_scan():
    """Clustered latents, which is the case Sec. IV assumes. Isotropic noise in
    147 dimensions has no near neighbours to find and the tree cannot help."""
    base = latents(15, n=64)
    jitter = 0.05 * jax.random.normal(jax.random.key(19), (64, 8) + SHAPE)
    s = (base[:, None] + jitter).reshape((512,) + SHAPE)
    flat = s.reshape(s.shape[0], -1)
    tree = SimilarityTree(flat, branching=4, leaf_size=32)
    q = s[300] + 0.01 * jax.random.normal(jax.random.key(16), SHAPE)
    direct = generate(s, q, k=3, sigma=0.5, eta=0.0)
    via_tree = generate(s, q, k=3, sigma=0.5, eta=0.0, tree=tree, max_leaves=8)
    assert sorted(via_tree.neighbor_indices) == sorted(direct.neighbor_indices)
    assert jnp.allclose(via_tree.latent, direct.latent, atol=1e-10)


def test_alg1_shapes_and_dtypes():
    s = latents(17, n=64)
    out = generate(s, s[0], k=3, sigma=0.5, eta=0.01, key=jax.random.key(18))
    assert out.image.shape == SHAPE
    assert out.weights.shape == (3,) + SHAPE[1:]
    assert out.neighbor_indices.shape == (3,)
    assert jnp.isfinite(out.latent).all()
