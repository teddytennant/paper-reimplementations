import jax.numpy as jnp
import numpy as np
import pytest

from cadmp.attention import attention_distributions
from cadmp.drift import drift_profile, select_mask_layer
from cadmp.features import extract, extract_drift, feature_dim
from cadmp.masking import GRAY, mask_image


def _attn(rng, layers=32, heads=4, tokens=256):
    raw = rng.random((layers, heads, tokens)) ** 3 + 1e-3
    raw = raw / raw.sum(axis=-1, keepdims=True) * 0.4
    return jnp.asarray(raw)


def test_feature_dim_is_l_plus_two():
    assert feature_dim(32) == 34  # LLaVA-1.5-7B: 31 drift values plus 3 probabilities
    assert feature_dim(28) == 30


def test_extract_lays_the_vector_out_as_drift_then_three_probabilities():
    rng = np.random.default_rng(0)
    attn = _attn(rng)
    out = extract(attn, 0.82, 0.31)

    assert out.features.shape == (feature_dim(32),)
    np.testing.assert_allclose(np.asarray(out.features[:31]), np.asarray(out.drift), rtol=1e-12)
    assert float(out.features[-3]) == pytest.approx(0.82)
    assert float(out.features[-2]) == pytest.approx(0.31)
    assert float(out.features[-1]) == pytest.approx(0.82 - 0.31)


def test_delta_p_is_positive_when_masking_costs_confidence():
    rng = np.random.default_rng(1)
    attn = _attn(rng, layers=8)
    assert float(extract(attn, 0.9, 0.2).features[-1]) > 0
    assert float(extract(attn, 0.2, 0.9).features[-1]) < 0


def test_extract_agrees_with_the_pieces_it_is_made_of():
    rng = np.random.default_rng(2)
    attn = _attn(rng, layers=12, heads=6, tokens=64)

    dists, drift, layer = extract_drift(attn)
    np.testing.assert_allclose(np.asarray(dists), np.asarray(attention_distributions(attn)), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(drift), np.asarray(drift_profile(dists)), rtol=1e-12)
    assert layer == select_mask_layer(drift)
    np.testing.assert_allclose(
        np.asarray(extract(attn, 0.5, 0.4).drift), np.asarray(drift), rtol=1e-12
    )


def test_the_documented_caller_workflow_runs():
    """The five steps in ``extract``'s docstring, with the LVLM stubbed out."""
    rng = np.random.default_rng(3)
    attn = _attn(rng, layers=32, heads=4, tokens=576)
    image = jnp.asarray(rng.uniform(0, 255, size=(336, 336, 3)))

    dists, _, layer = extract_drift(attn)
    masked = mask_image(image, dists[layer])

    assert 0 <= layer <= 30
    assert masked.shape == image.shape
    grayed = np.all(np.asarray(masked) == GRAY, axis=-1)
    assert grayed.mean() == pytest.approx(0.30, abs=0.01)

    out = extract(attn, 0.77, 0.41)
    assert out.features.shape == (34,)
    assert np.all(np.isfinite(np.asarray(out.features)))


def test_number_of_layers_is_not_hardcoded():
    rng = np.random.default_rng(4)
    for layers in (4, 17, 28, 32, 40):
        out = extract(_attn(rng, layers=layers, tokens=36), 0.5, 0.4)
        assert out.features.shape == (feature_dim(layers),)
        assert out.drift.shape == (layers - 1,)
