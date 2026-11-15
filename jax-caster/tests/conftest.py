"""Shared setup. x64 goes on before anything else touches jax.

Every quantity CASTER computes is a difference evaluated by cancellation:
Mahalanobis distances, matrix square roots, scatter matrices. On an NVIDIA GPU
float32 matmuls run at TF32 by default and lose around 1e-2 relative, which is
larger than the gaps these tests measure, so the whole suite runs in float64.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from caster import Caster, CasterConfig  # noqa: E402
from caster.synthetic import (  # noqa: E402
    affine_shift,
    class_parameters,
    random_spd,
    sample_classes,
)


def accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


def shifted_problem(seed, num_classes, dim, n_per_class, separation=3.0, translation=3.0,
                    config=None):
    """Source features, target features under a class-shared affine shift, and a fit.

    Source and target share one class geometry, so the target really is
    `A_true s + b_true` in distribution and Sec. 3.1's assumption holds.
    """
    config = config or CasterConfig()
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    params = class_parameters(keys[0], num_classes, dim, separation=separation, condition=4.0)
    source, source_labels = sample_classes(keys[1], params, n_per_class)
    target_clean, target_labels = sample_classes(keys[2], params, n_per_class)
    matrix = random_spd(keys[3], dim, condition=4.0)
    offset = jax.random.normal(keys[4], (dim,), dtype=source.dtype) * translation
    target = affine_shift(target_clean, matrix, offset)
    model = Caster.fit(source, source_labels, num_classes, config)
    return model, target, np.asarray(target_labels), matrix, offset


@pytest.fixture(scope="session")
def many_class_problem():
    """C = 30 in d = 24, the regime where gamma_s is well below the typical gap.

    That regime is what makes eq. (9) informative: with only a handful of
    well-spread classes the minimum gap is close to the typical gap and r_t
    cannot get much above 1 for any batch.
    """
    return shifted_problem(0, num_classes=30, dim=24, n_per_class=60, separation=2.0)


def batches_of(rng, labels, batch_size, count, classes=None):
    """Index batches, either i.i.d. or drawn from a fixed subset of classes."""
    labels = np.asarray(labels)
    out = []
    if classes is None:
        order = rng.permutation(len(labels))
        for i in range(count):
            out.append(order[i * batch_size:(i + 1) * batch_size])
    else:
        for _ in range(count):
            chosen = rng.choice(len(np.unique(labels)), classes, replace=False)
            pool = np.where(np.isin(labels, chosen))[0]
            out.append(rng.choice(pool, min(batch_size, len(pool)), replace=False))
    return [b for b in out if len(b) == batch_size]
