import jax.numpy as jnp
import numpy as np
import pytest

from cadmp.attention import attention_distributions, head_average, normalize


def _attn(rng, layers=6, heads=5, tokens=17, visual_mass=0.4):
    """Attention that behaves like the real thing: rows sum to less than 1.

    A decoder softmax runs over the whole prefix, so the visual tokens hold only
    part of the mass. Scaling by ``visual_mass`` reproduces that.
    """
    raw = rng.random((layers, heads, tokens))
    raw = raw / raw.sum(axis=-1, keepdims=True) * visual_mass
    return jnp.asarray(raw)


def test_head_average_matches_an_explicit_loop():
    rng = np.random.default_rng(0)
    attn = _attn(rng)
    layers, heads, tokens = attn.shape

    expected = np.zeros((layers, tokens))
    for i in range(layers):
        for m in range(tokens):
            expected[i, m] = sum(float(attn[i, h, m]) for h in range(heads)) / heads

    np.testing.assert_allclose(np.asarray(head_average(attn)), expected, rtol=0, atol=1e-12)


def test_head_average_is_a_mean_not_a_sum():
    attn = jnp.ones((3, 4, 5))
    np.testing.assert_allclose(np.asarray(head_average(attn)), np.ones((3, 5)))


def test_head_average_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[L, H, N\]"):
        head_average(jnp.ones((4, 7)))


def test_distributions_sum_to_one_over_visual_tokens():
    rng = np.random.default_rng(1)
    dists = attention_distributions(_attn(rng, visual_mass=0.23))
    np.testing.assert_allclose(np.asarray(dists.sum(axis=-1)), 1.0, atol=1e-12)


def test_normalization_is_scale_invariant_per_layer():
    """Only the shape of a layer's attention matters, not how much mass it holds."""
    rng = np.random.default_rng(2)
    attn = _attn(rng, visual_mass=1.0)
    scales = jnp.asarray(rng.uniform(0.05, 3.0, size=(attn.shape[0], 1, 1)))
    np.testing.assert_allclose(
        np.asarray(attention_distributions(attn)),
        np.asarray(attention_distributions(attn * scales)),
        atol=1e-12,
    )


def test_normalize_does_not_divide_by_n():
    """A uniform row must come out uniform, not scaled by 1/N twice."""
    a = jnp.full((1, 8), 0.25)
    np.testing.assert_allclose(np.asarray(normalize(a)), np.full((1, 8), 1 / 8), atol=1e-12)


def test_normalize_survives_an_all_zero_row():
    out = normalize(jnp.zeros((2, 6)))
    assert np.all(np.isfinite(np.asarray(out)))
