"""End to end on synthetic traces.

The data here is generated, not measured, so these tests check that the pipeline
recovers structure that was put in on purpose. They say nothing about CADMP's
performance on MSCOCO, POPE, AMBER or Pascal VOC.
"""

import jax
import numpy as np
import pytest

from cadmp.demo import evaluate, split
from cadmp.features import feature_dim
from cadmp.synthetic import generate

SEED = 42
N = 2000


@pytest.fixture(scope="module")
def data():
    return generate(n_samples=N, seed=SEED)


@pytest.fixture(scope="module")
def splits(data):
    return split(len(data.labels), seed=SEED)


def test_shapes_and_class_balance(data):
    assert data.features.shape == (N, feature_dim(32))
    assert data.drift.shape == (N, 31)
    assert set(np.unique(data.labels)) == {0, 1}
    assert 0.25 < data.labels.mean() < 0.35
    assert np.all(np.isfinite(data.features))


def test_generation_is_deterministic():
    a = generate(n_samples=200, seed=3)
    b = generate(n_samples=200, seed=3)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_hallucinated_samples_drift_more_in_the_middle_layers(data):
    """Figure 3's claim: the curves separate around layers 8 to 16."""
    grounded = data.drift[data.labels == 0].mean(axis=0)
    hallucinated = data.drift[data.labels == 1].mean(axis=0)

    # drift index j holds K_{j+2}, so paper layers 8..16 are indices 6..14
    middle = slice(6, 15)
    assert np.all(hallucinated[middle] > grounded[middle])
    assert hallucinated[middle].mean() > 1.5 * grounded[middle].mean()


def test_the_drift_gap_is_concentrated_in_the_middle(data):
    grounded = data.drift[data.labels == 0].mean(axis=0)
    hallucinated = data.drift[data.labels == 1].mean(axis=0)
    gap = hallucinated - grounded

    edges = np.r_[gap[:4], gap[-4:]]
    assert gap[6:15].mean() > 5 * edges.mean()


def test_peak_drift_lands_in_the_middle_layers_for_hallucinated_samples(data):
    """Figure 6's claim, on Qwen2.5-VL-7B the mode sits at layers 15, 10, 13."""
    peaks = data.mask_layer[data.labels == 1]
    assert np.median(peaks) == pytest.approx(11.0, abs=3.0)


def test_grounded_samples_lose_more_probability_after_masking(data):
    delta_p = data.features[:, -1]
    assert delta_p[data.labels == 0].mean() > delta_p[data.labels == 1].mean()
    assert delta_p[data.labels == 0].mean() > 1.8 * delta_p[data.labels == 1].mean()


def test_grounded_masks_cover_more_of_the_object_region(data):
    """Grounded attention sits on the object, so the mask hits it more often."""
    assert data.coverage[data.labels == 0].mean() > data.coverage[data.labels == 1].mean()


def test_probabilities_stay_in_range(data):
    p_orig, p_masked = data.features[:, -3], data.features[:, -2]
    assert p_orig.min() > 0.0 and p_orig.max() <= 1.0
    assert p_masked.min() > 0.0 and p_masked.max() <= 1.0


def test_detector_separates_held_out_synthetic_samples(data, splits):
    tr, va, te = splits
    au, ac = evaluate(
        data.features, data.labels, tr, va, te, jax.random.PRNGKey(SEED)
    )
    assert au > 0.80
    assert ac > 0.70


def test_combined_features_beat_either_half(data, splits):
    """Table 5's claim, reproduced on synthetic data only.

    The margin over drift alone is real but small, roughly 0.01 to 0.03 AUROC
    across generator seeds, so this is asserted at one fixed seed rather than as
    a general property of the generator.
    """
    tr, va, te = splits
    keys = jax.random.split(jax.random.PRNGKey(SEED), 3)
    n_layers = data.drift.shape[1] + 1

    drift_only, _ = evaluate(
        data.features[:, : n_layers - 1], data.labels, tr, va, te, keys[0]
    )
    mask_only, _ = evaluate(
        data.features[:, n_layers - 1 :], data.labels, tr, va, te, keys[1]
    )
    combined, _ = evaluate(data.features, data.labels, tr, va, te, keys[2])

    assert drift_only > 0.5 and mask_only > 0.5
    assert combined > drift_only
    assert combined > mask_only
