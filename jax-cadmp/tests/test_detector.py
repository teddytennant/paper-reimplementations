import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cadmp.detector import (
    accuracy,
    auroc,
    best_f1_threshold,
    class_weights,
    forward,
    init_params,
    loss_fn,
    weighted_sample,
)


def _pairwise_auroc(scores, labels):
    """Brute-force AUROC: probability a random positive outranks a random negative.

    Independent of the rank identity the implementation uses, so it catches a
    mistake in the tie handling rather than agreeing with it.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1][:, None]
    neg = scores[labels == 0][None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def test_auroc_matches_a_pairwise_oracle():
    rng = np.random.default_rng(0)
    for _ in range(10):
        labels = rng.integers(0, 2, size=200)
        scores = rng.normal(labels * 0.8, 1.0)
        assert auroc(scores, labels) == pytest.approx(_pairwise_auroc(scores, labels), rel=1e-12)


def test_auroc_handles_ties_with_midranks():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=300)
    scores = np.round(rng.normal(labels * 0.5, 1.0), 1)  # forces many ties
    assert len(np.unique(scores)) < 60
    assert auroc(scores, labels) == pytest.approx(_pairwise_auroc(scores, labels), rel=1e-12)


def test_auroc_edges():
    labels = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(0.0)
    assert auroc(np.full(4, 0.5), labels) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both classes"):
        auroc(np.array([0.1, 0.2]), np.array([1, 1]))


def test_accuracy():
    assert accuracy(np.array([0.9, 0.1, 0.6]), np.array([1, 0, 0]), 0.5) == pytest.approx(2 / 3)
    assert accuracy(np.array([0.9, 0.1, 0.6]), np.array([1, 0, 0]), 0.7) == pytest.approx(1.0)


def test_best_f1_threshold_finds_the_separating_cut():
    scores = np.array([0.05, 0.1, 0.15, 0.8, 0.85, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = best_f1_threshold(scores, labels)
    assert 0.15 < t <= 0.8
    assert accuracy(scores, labels, t) == pytest.approx(1.0)


def test_class_weights_are_inverse_class_frequency():
    labels = np.array([0] * 7 + [1] * 3)
    w = np.asarray(class_weights(labels))
    np.testing.assert_allclose(w[labels == 0], 1 / 7)
    np.testing.assert_allclose(w[labels == 1], 1 / 3)
    # each class contributes the same total mass, which is the point
    assert w[labels == 0].sum() == pytest.approx(w[labels == 1].sum())


def test_weighted_sampling_balances_a_skewed_pool():
    """900 grounded against 100 hallucinated should still draw 50/50."""
    labels = np.array([0] * 900 + [1] * 100)
    draws = 200_000
    idx = np.asarray(weighted_sample(jax.random.PRNGKey(0), class_weights(labels), draws))
    frequency = labels[idx].mean()
    # binomial standard error at p=0.5 over 200k draws is ~0.0011
    assert abs(frequency - 0.5) < 0.005


def test_weighted_sampling_reaches_every_minority_sample():
    labels = np.array([0] * 500 + [1] * 5)
    idx = np.asarray(weighted_sample(jax.random.PRNGKey(1), class_weights(labels), 50_000))
    assert set(np.arange(500, 505)).issubset(set(idx.tolist()))


def test_weighted_sampling_is_uniform_within_a_balanced_pool():
    labels = np.zeros(10, dtype=int)
    idx = np.asarray(weighted_sample(jax.random.PRNGKey(2), class_weights(labels), 100_000))
    counts = np.bincount(idx, minlength=10) / 100_000
    assert np.max(np.abs(counts - 0.1)) < 0.005


def _flatten(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    return leaves, treedef


def test_loss_gradient_matches_central_finite_differences():
    """Checks the whole forward pass, batch norm and all, not just the last layer."""
    key = jax.random.PRNGKey(0)
    params, state = init_params(key, in_dim=34, hidden=(8, 5))

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(24, 34)))
    y = jnp.asarray(rng.integers(0, 2, size=24).astype(float))

    def scalar_loss(p):
        # dropout off, so the function is deterministic and differentiable
        return loss_fn(p, state, x, y, True, None, 0.0)[0]

    grads = jax.grad(scalar_loss)(params)
    leaves, treedef = _flatten(params)
    grad_leaves, _ = _flatten(grads)

    checked = 0
    step = 1e-6
    for slot, (leaf, gleaf) in enumerate(zip(leaves, grad_leaves)):
        flat = leaf.reshape(-1)
        picks = rng.choice(flat.shape[0], size=min(3, flat.shape[0]), replace=False)
        for pick in picks:
            def bumped(delta):
                new = flat.at[pick].add(delta).reshape(leaf.shape)
                rebuilt = list(leaves)
                rebuilt[slot] = new
                return scalar_loss(jax.tree_util.tree_unflatten(treedef, rebuilt))

            numeric = (float(bumped(step)) - float(bumped(-step))) / (2 * step)
            analytic = float(gleaf.reshape(-1)[pick])
            assert numeric == pytest.approx(analytic, rel=2e-5, abs=1e-8)
            checked += 1

    assert checked >= 15


def test_dropout_is_off_at_inference_and_on_during_training():
    key = jax.random.PRNGKey(3)
    params, state = init_params(key, in_dim=6, hidden=(16, 8))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(32, 6)))

    a, _ = forward(params, state, x, train=False)
    b, _ = forward(params, state, x, train=False)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    c, _ = forward(params, state, x, train=True, key=jax.random.PRNGKey(4))
    d, _ = forward(params, state, x, train=True, key=jax.random.PRNGKey(5))
    assert not np.allclose(np.asarray(c), np.asarray(d))


def test_batch_norm_running_statistics_move_only_in_training():
    key = jax.random.PRNGKey(6)
    params, state = init_params(key, in_dim=4, hidden=(5,))
    x = jnp.asarray(np.random.default_rng(0).normal(3.0, 2.0, size=(64, 4)))

    _, unchanged = forward(params, state, x, train=False)
    np.testing.assert_array_equal(np.asarray(unchanged[0]["mean"]), np.asarray(state[0]["mean"]))

    _, updated = forward(params, state, x, train=True)
    assert not np.allclose(np.asarray(updated[0]["mean"]), np.asarray(state[0]["mean"]))


def test_output_is_a_single_logit_per_sample():
    params, state = init_params(jax.random.PRNGKey(7), in_dim=34)
    logits, _ = forward(params, state, jnp.zeros((11, 34)), train=False)
    assert logits.shape == (11,)
    probs = jax.nn.sigmoid(logits)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0
