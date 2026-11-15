"""A runnable version of Sec. 4.4's three policies on synthetic features.

Fits CASTER on 30 Gaussian classes in 24 dimensions, then adapts two kinds of
target batch under the same class-shared affine shift:

  i.i.d.     256 samples drawn from all 30 classes, the regime the transport is
             built for;
  few-class  256 samples drawn from 3 classes, the scaled-down version of
             Sec. 4.6's 64 samples against 1000 classes.

The three policies are never transport (the Source-GDA fallback), always
transport, and transport only when the certificate of eq. (9) accepts.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .config import CasterConfig  # noqa: E402
from .estimator import Caster  # noqa: E402
from .synthetic import affine_shift, class_parameters, random_spd, sample_classes  # noqa: E402


def _accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


def _batches(rng, labels, batch_size, count, classes=None):
    labels = np.asarray(labels)
    if classes is None:
        order = rng.permutation(len(labels))
        return [order[i * batch_size:(i + 1) * batch_size] for i in range(count)]
    out = []
    for _ in range(count):
        chosen = rng.choice(len(np.unique(labels)), classes, replace=False)
        pool = np.where(np.isin(labels, chosen))[0]
        out.append(rng.choice(pool, batch_size, replace=False))
    return out


def _policies(model, target, labels, selections):
    rows = []
    for sel in selections:
        result = model.adapt(target[sel])
        rows.append((
            _accuracy(result.fallback, labels[sel]),
            _accuracy(result.transported, labels[sel]),
            _accuracy(result.predictions, labels[sel]),
            result.certificate,
            result.accepted,
        ))
    columns = list(zip(*rows))
    return [float(np.mean(c)) for c in columns[:3]] + [
        float(np.median(columns[3])), float(np.mean(columns[4]))
    ]


def main(seed=0, num_classes=30, dim=24, n_per_class=120):
    config = CasterConfig()
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    params = class_parameters(keys[0], num_classes, dim, separation=2.0, condition=4.0)
    source, source_labels = sample_classes(keys[1], params, n_per_class)
    clean, labels = sample_classes(keys[2], params, n_per_class)
    target = affine_shift(
        clean,
        random_spd(keys[3], dim, condition=4.0),
        jax.random.normal(keys[4], (dim,), dtype=jnp.float64) * 3.0,
    )
    labels = np.asarray(labels)

    model = Caster.fit(source, source_labels, num_classes, config)
    rng = np.random.default_rng(seed)

    print(f"{num_classes} classes, {dim} dims, k0 = {model.state.k0}, tau = {config.tau}")
    print(f"{'batch':<12}{'never':>8}{'always':>8}{'gated':>8}{'r_t':>8}{'accept':>8}")
    for name, selections in [
        ("i.i.d.", _batches(rng, labels, 256, 6)),
        ("3 classes", _batches(rng, labels, 256, 6, classes=3)),
    ]:
        never, always, gated, cert, accept = _policies(model, target, labels, selections)
        print(f"{name:<12}{never:>8.3f}{always:>8.3f}{gated:>8.3f}{cert:>8.2f}{accept:>8.2f}")


if __name__ == "__main__":
    main()
