"""The lightweight MLP detector, its training loop, and the two metrics.

Architecture, from the appendix's implementation details: standardize the
features, then two hidden layers of 128 and 64 units, each followed by batch
normalization, ReLU and dropout at rate 0.3, then a linear output. The paper
says the detector "outputs a hallucination score for each object prediction,
which is converted into a probability through a sigmoid function", so the output
is a single logit and the loss is binary cross-entropy. Optimizer AdamW, lr
5e-4, weight decay 1e-4, batch size 128, at most 150 epochs, early stopping with
patience 20 on the validation set.

Class imbalance is handled the way the paper handles it, with weighted random
sampling: "assigning each sample a weight inversely proportional to the size of
its class". So a batch is drawn with replacement from a categorical distribution
whose weight on sample ``k`` is ``1 / count(class of k)``, which makes the two
classes equally likely per draw regardless of how skewed the pool is. No loss
reweighting on top; that would double-count.

Metrics are AUROC and accuracy, the paper's two, with hallucinated as the
positive class.

Three things the paper leaves open and this file has to pick:

* Which quantity early stopping watches. The appendix says "the best checkpoint
  on the validation set" without naming a metric. Validation AUROC is used here,
  since it is the headline metric and does not depend on a threshold.
* Whether weight decay skips the batch-norm scales and the biases. Common
  practice says it should; the paper does not say, so ``optax.adamw`` is applied
  to every parameter, which is the plainer reading of "AdamW with a weight decay
  of 1e-4".
* Batch-norm momentum. Not stated. 0.1 is used, matching the PyTorch default the
  authors would have inherited.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

__all__ = [
    "Detector",
    "StandardScaler",
    "init_params",
    "forward",
    "loss_fn",
    "class_weights",
    "weighted_sample",
    "auroc",
    "accuracy",
    "best_f1_threshold",
    "train",
    "predict",
]


class StandardScaler(NamedTuple):
    """Per-feature mean and standard deviation, fit on the training split."""

    mean: jnp.ndarray
    std: jnp.ndarray

    @staticmethod
    def fit(x: jnp.ndarray, eps: float = 1e-8) -> "StandardScaler":
        x = jnp.asarray(x)
        return StandardScaler(
            mean=jnp.mean(x, axis=0), std=jnp.maximum(jnp.std(x, axis=0), eps)
        )

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(x) - self.mean) / self.std


class Detector(NamedTuple):
    """A trained detector: parameters, frozen batch-norm statistics, scaler."""

    params: list
    state: list
    scaler: StandardScaler
    threshold: float
    """Decision threshold on the sigmoid output, fit on the validation split."""


def init_params(key, in_dim: int, hidden=(128, 64)):
    """He-initialized weights, zero biases, unit batch-norm scales.

    Returns ``(params, state)``. ``params`` is a list with one dict per hidden
    block (``w``, ``b``, ``gamma``, ``beta``) plus a final dict (``w``, ``b``)
    for the output logit. ``state`` holds the batch-norm running statistics, one
    dict per hidden block.
    """
    params, state = [], []
    dims = (in_dim,) + tuple(hidden)
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        key, subkey = jax.random.split(key)
        scale = jnp.sqrt(2.0 / fan_in)
        params.append(
            {
                "w": jax.random.normal(subkey, (fan_in, fan_out)) * scale,
                "b": jnp.zeros((fan_out,)),
                "gamma": jnp.ones((fan_out,)),
                "beta": jnp.zeros((fan_out,)),
            }
        )
        state.append({"mean": jnp.zeros((fan_out,)), "var": jnp.ones((fan_out,))})
    key, subkey = jax.random.split(key)
    params.append(
        {
            "w": jax.random.normal(subkey, (dims[-1], 1)) * jnp.sqrt(2.0 / dims[-1]),
            "b": jnp.zeros((1,)),
        }
    )
    return params, state


def forward(
    params,
    state,
    x: jnp.ndarray,
    train: bool = False,
    key=None,
    dropout_rate: float = 0.3,
    momentum: float = 0.1,
    bn_eps: float = 1e-5,
):
    """Run the MLP and return ``(logits, new_state)``.

    In training mode each block normalizes by the batch's own mean and variance
    and folds them into the running statistics; in inference mode it uses the
    running statistics and returns ``state`` unchanged. Dropout is applied only
    in training mode and only when ``key`` is not None, so passing ``key=None``
    gives the deterministic function that the finite-difference gradient check
    in ``tests/test_detector.py`` differentiates.

    Args:
        params: from ``init_params``.
        state: batch-norm running statistics.
        x: ``[B, D]`` standardized features.
        train: batch statistics and dropout on, or running statistics and off.
        key: PRNG key for dropout.
        dropout_rate: 0.3 in the paper.
        momentum: weight on the batch statistic in the running average.
        bn_eps: variance floor inside the normalization.

    Returns:
        ``(logits, new_state)`` with logits of shape ``[B]``.
    """
    h = jnp.asarray(x)
    new_state = []
    for layer, stats in zip(params[:-1], state):
        h = h @ layer["w"] + layer["b"]
        if train:
            mean = jnp.mean(h, axis=0)
            var = jnp.var(h, axis=0)
            new_state.append(
                {
                    "mean": (1 - momentum) * stats["mean"] + momentum * mean,
                    "var": (1 - momentum) * stats["var"] + momentum * var,
                }
            )
        else:
            mean, var = stats["mean"], stats["var"]
            new_state.append(stats)
        h = (h - mean) / jnp.sqrt(var + bn_eps)
        h = layer["gamma"] * h + layer["beta"]
        h = jax.nn.relu(h)
        if train and key is not None and dropout_rate > 0.0:
            key, subkey = jax.random.split(key)
            keep = 1.0 - dropout_rate
            h = h * jax.random.bernoulli(subkey, keep, h.shape) / keep
    logits = (h @ params[-1]["w"] + params[-1]["b"])[:, 0]
    return logits, new_state


def loss_fn(
    params,
    state,
    x: jnp.ndarray,
    y: jnp.ndarray,
    train: bool = True,
    key=None,
    dropout_rate: float = 0.3,
):
    """Mean binary cross-entropy on the sigmoid output.

    Labels are 1 for hallucinated, 0 for grounded. Written through
    ``optax.sigmoid_binary_cross_entropy`` so it stays stable at large |logit|.

    Returns ``(loss, new_state)``.
    """
    logits, new_state = forward(
        params, state, x, train=train, key=key, dropout_rate=dropout_rate
    )
    losses = optax.sigmoid_binary_cross_entropy(logits, jnp.asarray(y, logits.dtype))
    return jnp.mean(losses), new_state


def class_weights(labels: jnp.ndarray) -> jnp.ndarray:
    """Per-sample sampling weight, inversely proportional to its class size.

    ``w_k = 1 / count(class of k)``, the paper's rule. Summed over a class these
    weights give 1 no matter how many members it has, so drawing with
    replacement from ``w / sum(w)`` puts equal probability on each class.

    Args:
        labels: ``[n]`` of 0/1.

    Returns:
        ``[n]`` unnormalized weights.
    """
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(float)
    if np.any(counts[np.unique(labels)] == 0):
        raise ValueError("a present class has zero count")
    return jnp.asarray(1.0 / counts[labels])


def weighted_sample(key, weights: jnp.ndarray, size: int) -> jnp.ndarray:
    """Draw ``size`` indices with replacement, proportional to ``weights``.

    This is the whole of "weighted random sampling" from the paper: PyTorch's
    ``WeightedRandomSampler`` with ``replacement=True`` draws from exactly this
    categorical distribution.
    """
    weights = jnp.asarray(weights)
    logits = jnp.log(weights) - jnp.log(jnp.sum(weights))
    return jax.random.categorical(key, logits, shape=(size,))


def auroc(scores: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Area under the ROC curve, hallucinated (label 1) as the positive class.

    Computed by the Mann-Whitney rank identity,

        AUROC = (sum of positive ranks - n_pos (n_pos + 1) / 2) / (n_pos n_neg),

    with midranks for ties, so a detector that outputs a constant scores 0.5
    rather than 0 or 1.
    """
    scores = jnp.asarray(scores, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    labels = jnp.asarray(labels)
    n_pos = jnp.sum(labels == 1)
    n_neg = jnp.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")
    ordered = jnp.sort(scores)
    lo = jnp.searchsorted(ordered, scores, side="left")
    hi = jnp.searchsorted(ordered, scores, side="right")
    ranks = (lo + hi + 1) / 2.0
    pos_rank_sum = jnp.sum(jnp.where(labels == 1, ranks, 0.0))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy(scores: jnp.ndarray, labels: jnp.ndarray, threshold: float = 0.5) -> float:
    """Fraction correct after thresholding the sigmoid output."""
    pred = jnp.asarray(scores) >= threshold
    return float(jnp.mean(pred == (jnp.asarray(labels) == 1)))


def best_f1_threshold(scores: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Threshold on the hallucination score that maximizes F1 on this split.

    The appendix says the threshold is "selected on the validation set by
    maximizing the F1 score", computed from "the probability of the grounded
    class". With one sigmoid output the grounded probability is ``1 - score``,
    so the only real content of that sentence is which class F1 is measured on.
    This maximizes F1 for the hallucinated class, the same class AUROC treats as
    positive, and picks the lowest threshold among ties.

    Candidates are the observed scores themselves, which is enough because the
    prediction only changes as the threshold crosses one of them.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    candidates = np.unique(scores)
    best_f1, best_t = -1.0, 0.5
    for t in candidates:
        pred = scores >= t
        tp = float(np.sum(pred & (labels == 1)))
        fp = float(np.sum(pred & (labels == 0)))
        fn = float(np.sum(~pred & (labels == 1)))
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else 2 * tp / denom
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def predict(detector: Detector, x: jnp.ndarray) -> jnp.ndarray:
    """Hallucination probabilities for raw (unstandardized) features."""
    logits, _ = forward(
        detector.params, detector.state, detector.scaler.transform(x), train=False
    )
    return jax.nn.sigmoid(logits)


def train(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    key,
    hidden=(128, 64),
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    max_epochs: int = 150,
    patience: int = 20,
    dropout_rate: float = 0.3,
) -> Detector:
    """Fit the detector, keeping the checkpoint with the best validation AUROC.

    Defaults are the paper's appendix values. Batches are drawn with replacement
    under the inverse-class-frequency weights, so one "epoch" here is
    ``ceil(n_train / batch_size)`` such draws rather than a pass over each
    sample exactly once.

    Args:
        x_train, y_train: raw features and 0/1 labels for the training split.
        x_val, y_val: same for validation; drives early stopping and the
            decision threshold.
        key: PRNG key.
        hidden: hidden widths, (128, 64) in the paper.
        lr, weight_decay, batch_size, max_epochs, patience, dropout_rate: as in
            the appendix.

    Returns:
        A ``Detector`` holding the best-validation-AUROC parameters.
    """
    x_train = jnp.asarray(x_train)
    x_val = jnp.asarray(x_val)
    y_train = jnp.asarray(y_train)
    y_val = jnp.asarray(y_val)

    scaler = StandardScaler.fit(x_train)
    xt = scaler.transform(x_train)
    xv = scaler.transform(x_val)

    key, init_key = jax.random.split(key)
    params, state = init_params(init_key, xt.shape[1], hidden=hidden)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)
    weights = class_weights(y_train)

    @jax.jit
    def step(params, state, opt_state, xb, yb, step_key):
        (_, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, xb, yb, True, step_key, dropout_rate
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, opt_state

    @jax.jit
    def val_scores(params, state, x):
        logits, _ = forward(params, state, x, train=False)
        return jax.nn.sigmoid(logits)

    steps_per_epoch = max(1, -(-xt.shape[0] // batch_size))
    best = (-1.0, params, state)
    since_improved = 0
    for _ in range(max_epochs):
        for _ in range(steps_per_epoch):
            key, idx_key, drop_key = jax.random.split(key, 3)
            idx = weighted_sample(idx_key, weights, batch_size)
            params, state, opt_state = step(
                params, state, opt_state, xt[idx], y_train[idx], drop_key
            )
        score = auroc(val_scores(params, state, xv), y_val)
        if score > best[0]:
            best = (score, params, state)
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    _, params, state = best
    threshold = best_f1_threshold(val_scores(params, state, xv), y_val)
    return Detector(params=params, state=state, scaler=scaler, threshold=threshold)
