"""CASTER end to end: does the transport help, and does the gate catch it when it does not."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from caster import Caster, CasterConfig  # noqa: E402
from caster.synthetic import class_parameters, sample_classes  # noqa: E402

from conftest import accuracy, batches_of, shifted_problem  # noqa: E402


def test_transport_recovers_oracle_accuracy_under_an_affine_shift():
    """Under a true class-shared affine shift, transport lands on the oracle.

    The oracle is a Gaussian discriminant fit on *labeled* target features. The
    comparison is the mean over three seeds, since a single seed of random
    Gaussians proves nothing; the per-seed gaps are all in the same direction
    anyway. The shift is a random SPD matrix with condition 4 plus a
    translation of about 3 units, which is large enough that the untransported
    source discriminant loses 10 to 25 points.
    """
    config = CasterConfig(lam=1e-4, beta=1e-4)
    transported, source_only, oracle_only = [], [], []

    for seed in range(3):
        model, target, labels, _, _ = shifted_problem(
            seed, num_classes=7, dim=6, n_per_class=400, config=config
        )
        oracle = Caster.fit(target, jnp.asarray(labels), 7, config)
        rng = np.random.default_rng(seed)
        for sel in batches_of(rng, labels, 256, 4):
            result = model.adapt(target[sel])
            transported.append(accuracy(result.transported, labels[sel]))
            source_only.append(accuracy(result.fallback, labels[sel]))
            oracle_only.append(accuracy(oracle.source_predictions(target[sel]), labels[sel]))

    transported, source_only, oracle_only = map(np.mean, (transported, source_only, oracle_only))
    assert transported > source_only + 0.08
    assert transported > oracle_only - 0.03


def test_certificate_is_small_on_iid_batches_and_large_on_few_class_batches(many_class_problem):
    """The degenerate regime of Sec. 4.6, scaled down.

    ImageNet-C puts 64 samples against 1000 classes. Here it is 128 samples
    drawn from 3 of 30 classes: the batch moments no longer describe the target
    distribution, so eq. (3) estimates a badly shrunk map and the transported
    predictions are worse than doing nothing. r_t rises with it. The paper is
    explicit (Sec. 4.3) that the benign and degenerate bands are not disjoint,
    so this compares the two medians rather than asserting a clean split.
    """
    model, target, labels, _, _ = many_class_problem
    rng = np.random.default_rng(0)

    iid_batches = batches_of(rng, labels, 256, 4)
    few_batches = batches_of(rng, labels, 128, 4, classes=3)
    iid = [model.adapt(target[sel]) for sel in iid_batches]
    few = [model.adapt(target[sel]) for sel in few_batches]

    assert np.median([r.certificate for r in few]) > 2.0 * np.median(
        [r.certificate for r in iid]
    )

    iid_gain = np.mean([
        accuracy(r.transported, labels[sel]) - accuracy(r.fallback, labels[sel])
        for r, sel in zip(iid, iid_batches)
    ])
    assert iid_gain > 0.10  # transport is worth having on the i.i.d. batches


def test_gate_rejects_the_degenerate_batches_and_returns_the_fallback(many_class_problem):
    """Every few-class batch is rejected, and rejection means the fallback verbatim."""
    model, target, labels, _, _ = many_class_problem
    rng = np.random.default_rng(0)

    for sel in batches_of(rng, labels, 128, 4, classes=3):
        result = model.adapt(target[sel])
        assert not result.accepted
        assert np.array_equal(np.asarray(result.predictions), np.asarray(result.fallback))
        assert accuracy(result.transported, labels[sel]) < accuracy(result.fallback, labels[sel])


def test_gating_does_not_lose_to_the_fallback_on_destructive_batches(many_class_problem):
    """The asymmetry Sec. 4.4 describes: rejecting costs a little, accepting a
    bad transport costs a lot. On batches where transport is destructive, gated
    accuracy has to stay within a point of the fallback."""
    model, target, labels, _, _ = many_class_problem
    rng = np.random.default_rng(1)

    gated, fallback = [], []
    for sel in batches_of(rng, labels, 128, 4, classes=3):
        result = model.adapt(target[sel])
        gated.append(accuracy(result.predictions, labels[sel]))
        fallback.append(accuracy(result.fallback, labels[sel]))
    assert np.mean(gated) >= np.mean(fallback) - 0.01


def test_gate_rejects_when_class_coverage_cannot_be_met():
    """Sec. 4.6's arithmetic: with 80 classes and a batch of 16, the retained
    set cannot occupy the required 20 pseudo-classes, so the batch is rejected
    by construction whatever r_t says."""
    config = CasterConfig()
    model, target, labels, _, _ = shifted_problem(
        3, num_classes=80, dim=24, n_per_class=20, separation=2.5, config=config
    )
    rng = np.random.default_rng(2)
    for sel in batches_of(rng, labels, 16, 5):
        result = model.adapt(target[sel])
        assert not result.accepted
        assert "classes" in result.decision.reason
        assert result.decision.certificate.n_occupied <= 16


def test_head_predictions_are_used_as_the_fallback_when_supplied():
    """The paper falls back to the frozen model head. Without a backbone the
    default is Source-GDA, but a head can be handed in."""
    model, target, labels, _, _ = shifted_problem(
        4, num_classes=30, dim=24, n_per_class=60, separation=2.0
    )
    rng = np.random.default_rng(3)
    sel = batches_of(rng, labels, 128, 1, classes=3)[0]
    head = jnp.zeros((len(sel),), dtype=jnp.int32)
    result = model.adapt(target[sel], head_predictions=head)
    assert not result.accepted
    assert np.array_equal(np.asarray(result.predictions), np.zeros(len(sel)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_input_dtype_is_preserved(dtype):
    """Nothing in the library forces float32 or float64 on the caller."""
    params = class_parameters(jax.random.PRNGKey(0), 6, 8, separation=3.0)
    features, labels = sample_classes(jax.random.PRNGKey(1), params, 60)
    features = features.astype(dtype)
    model = Caster.fit(features, labels, 6, CasterConfig())
    assert model.state.means.dtype == dtype
    assert model.state.pooled_cov.dtype == dtype
    result = model.adapt(features[:64])
    assert result.z.dtype == dtype
    assert result.affine.a.dtype == dtype
