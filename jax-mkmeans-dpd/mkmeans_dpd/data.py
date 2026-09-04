"""Datasets: the Section 4.2 simulation and the Iris data of Section 5.1."""

from pathlib import Path

import numpy as np

_IRIS = Path(__file__).with_name("iris.csv")

# Section 4.2: 750 points in 2 dimensions, 150 / 300 / 300 from three groups.
SIM_SIZES = (150, 300, 300)
SIM_MEANS = np.array([[6.0, 8.0], [11.0, 6.0], [0.0, 11.0]])
SIM_COVS = np.array(
    [
        [[2.0, 1.4], [1.4, 2.0]],
        [[2.0, -1.4], [-1.4, 2.0]],
        [[2.0, 0.0], [0.0, 2.0]],
    ]
)


def make_simulated(seed=0, n_outliers=30, pad=0.6):
    """The simulated dataset of Section 4.2, plus outliers.

    Group means (6, 8), (11, 6), (0, 11) with covariances that have diagonal 2
    and off-diagonal 1.4, -1.4 and 0 respectively. Returns (x, y) where y is the
    true group index and outliers carry label -1.

    The paper says only that "30 outliers were introduced into the dataset by
    appending them"; it does not say how they were drawn. We draw them uniformly
    over the clean data's bounding box grown by ``pad`` on each side and reject
    any that land within the 99% ellipsoid of a real group, so every outlier is
    genuinely outside the clusters. That choice is ours, not the paper's.
    """
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for j, (m, cov, size) in enumerate(zip(SIM_MEANS, SIM_COVS, SIM_SIZES)):
        parts.append(rng.multivariate_normal(m, cov, size=size))
        labels.append(np.full(size, j))
    x = np.concatenate(parts)
    y = np.concatenate(labels)

    lo, hi = x.min(0), x.max(0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    inv = np.linalg.inv(SIM_COVS)
    cutoff = 9.21034  # chi-square, 2 df, 99%

    out = []
    while len(out) < n_outliers:
        cand = rng.uniform(lo, hi, size=(n_outliers, 2))
        d = cand[:, None, :] - SIM_MEANS[None, :, :]
        b = np.einsum("nkj,kjl,nkl->nk", d, inv, d)
        out.extend(cand[b.min(1) > cutoff])
    out = np.asarray(out[:n_outliers])

    x = np.concatenate([x, out])
    y = np.concatenate([y, np.full(n_outliers, -1)])
    return x, y


def load_iris():
    """The Iris data of Section 5.1, 150 rows, 4 features, 3 species.

    Vendored as a CSV in the package. Returns (x, y) with y in {0, 1, 2} for
    setosa, versicolor, virginica.
    """
    raw = np.loadtxt(_IRIS, delimiter=",", skiprows=1)
    return raw[:, :4], raw[:, 4].astype(int)


def cluster_accuracy(y_true, labels, k=None):
    """Accuracy under the best matching of cluster ids to class labels.

    Equation (4.9) counts agreements; clustering does not fix the labelling, so
    we maximize over the K! permutations first. Points with y_true < 0 (the
    injected outliers) are excluded from the count.
    """
    from itertools import permutations

    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    keep = y_true >= 0
    y_true, labels = y_true[keep], labels[keep]
    k = int(max(y_true.max(), labels.max())) + 1 if k is None else k
    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.asarray(perm)[labels]
        best = max(best, float((mapped == y_true).mean()))
    return best
