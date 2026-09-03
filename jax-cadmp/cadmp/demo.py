"""End-to-end run on synthetic traces: generate, split, train, score.

Prints the drift curves that stand in for Figure 3, the detector's AUROC and
accuracy on a held-out test split, and the CMAD/MV ablation from Table 5. The
data is synthetic, so the numbers say the pipeline is wired correctly and
nothing about how CADMP does on MSCOCO.
"""

from __future__ import annotations

import jax
import numpy as np

from .detector import accuracy, auroc, predict, train
from .synthetic import generate


def split(n: int, seed: int = 42, ratios=(0.7, 0.1, 0.2)):
    """Shuffled 70/10/20 train, validation, test index split, as in the appendix."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(ratios[0] * n)
    b = a + int(ratios[1] * n)
    return idx[:a], idx[a:b], idx[b:]


def evaluate(features, labels, tr, va, te, key):
    """Train on one feature subset and report test AUROC and accuracy."""
    detector = train(features[tr], labels[tr], features[va], labels[va], key)
    scores = predict(detector, features[te])
    return (
        auroc(scores, labels[te]),
        accuracy(scores, labels[te], detector.threshold),
    )


def main(seed: int = 42, n_samples: int = 3000) -> None:
    data = generate(n_samples=n_samples, seed=seed)
    labels = data.labels
    n_layers = data.drift.shape[1] + 1
    tr, va, te = split(len(labels), seed=seed)

    print(f"{len(labels)} synthetic samples, L = {n_layers}, "
          f"features = {data.features.shape[1]}, "
          f"hallucinated = {labels.mean():.1%}")

    grounded = data.drift[labels == 0].mean(axis=0)
    hallucinated = data.drift[labels == 1].mean(axis=0)
    print("\nmean adjacent-layer drift, the Figure 3 stand-in")
    print("layer  grounded  hallucinated")
    for j in range(0, len(grounded), 2):
        print(f"{j + 2:5d}  {grounded[j]:8.4f}  {hallucinated[j]:12.4f}")

    print("\nmean delta_P after masking")
    dp = data.features[:, -1]
    print(f"  grounded      {dp[labels == 0].mean():.4f}")
    print(f"  hallucinated  {dp[labels == 1].mean():.4f}")

    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    drift_only = np.arange(n_layers - 1)
    mask_only = np.arange(n_layers - 1, n_layers + 2)

    print("\ntest set, hallucinated is the positive class")
    print("features          AUROC   ACC")
    for name, cols, key in (
        ("CMAD only", drift_only, keys[0]),
        ("MV only", mask_only, keys[1]),
        ("CADMP", np.arange(data.features.shape[1]), keys[2]),
    ):
        au, ac = evaluate(data.features[:, cols], labels, tr, va, te, key)
        print(f"{name:16s} {au:6.3f} {ac:5.3f}")


if __name__ == "__main__":
    main()
