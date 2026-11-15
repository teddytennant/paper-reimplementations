"""The finite population and its class summaries, Section 6.2 of arXiv:2609.04420.

A fold has n rows indexed by i, with labels Y_i in {0, 1}, N_1 = pi n positives and
N_0 = n - N_1 negatives. Equation (6) puts the class-weighted loss contribution of a row
at

    l_i = omega_{Y_i} * l(Phi(x_i), Y_i),    omega_1 = N_0 / N_1,   omega_0 = 1,

and the full-data objective is the total L(Phi) = sum_i l_i. Everything downstream is a
functional of the vector (l_i) and the labels, so a "population" here is exactly that
pair; the model state Phi is fixed throughout Section 6 and never appears.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

CLASSES = (0, 1)


class Population(NamedTuple):
    """A fold: weighted losses l_i of Equation (6) and their labels.

    ``losses`` are already multiplied by omega_{Y_i}. ``raw`` keeps the unweighted
    contributions l(Phi(x_i), Y_i) around, since Corollary 9 is stated in terms of their
    dispersions sigma_c and the relation S_1 = omega_1 sigma_1, S_0 = sigma_0.
    """

    losses: jax.Array
    labels: jax.Array
    raw: jax.Array

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    @property
    def total(self) -> jax.Array:
        """L(Phi) = sum_i l_i, the estimand of Section 6.2."""
        return jnp.sum(self.losses)

    @property
    def prevalence(self) -> jax.Array:
        """pi = N_1 / n."""
        counts = class_counts(self.labels)
        return counts[1] / jnp.sum(counts)


def class_weights(labels: jax.Array) -> jax.Array:
    """omega_1 = N_0 / N_1 and omega_0 = 1, Equation (6), as a per-row vector."""
    counts = class_counts(labels)
    omega1 = counts[0] / counts[1]
    return jnp.where(labels == 1, omega1, 1.0)


def weighted_losses(raw: jax.Array, labels: jax.Array) -> jax.Array:
    """Apply the class multiplier of Equation (6) to unweighted contributions."""
    return class_weights(labels) * raw


def class_counts(labels: jax.Array) -> jax.Array:
    """(N_0, N_1)."""
    labels = jnp.asarray(labels)
    n1 = jnp.sum(labels == 1)
    n0 = jnp.sum(labels == 0)
    return jnp.stack([n0, n1]).astype(jnp.float64)


def class_means(values: jax.Array, labels: jax.Array) -> jax.Array:
    """(lbar_0, lbar_1), the class-c means of the supplied per-row vector."""
    counts = class_counts(labels)
    sums = jnp.stack(
        [jnp.sum(jnp.where(labels == c, values, 0.0)) for c in CLASSES]
    )
    return sums / counts


def class_dispersions(values: jax.Array, labels: jax.Array) -> jax.Array:
    """S_c of Equation (7): the within-class dispersion with an N_c - 1 denominator.

    Applied to the weighted losses this is S_c; applied to the unweighted ones it is
    sigma_c, and S_1 = omega_1 sigma_1, S_0 = sigma_0.
    """
    counts = class_counts(labels)
    means = class_means(values, labels)
    sq = jnp.stack(
        [
            jnp.sum(jnp.where(labels == c, (values - means[c]) ** 2, 0.0))
            for c in CLASSES
        ]
    )
    return jnp.sqrt(sq / (counts - 1.0))


def pooled_dispersion(values: jax.Array) -> jax.Array:
    """S, the dispersion of l over all n rows, used by the uniform-design variance."""
    n = values.shape[0]
    return jnp.sqrt(jnp.sum((values - jnp.mean(values)) ** 2) / (n - 1.0))


def between_class_term(values: jax.Array, labels: jax.Array) -> jax.Array:
    """B = sum_c N_c (lbar_c - lbar)^2, the between-class term of Proposition 6."""
    counts = class_counts(labels)
    means = class_means(values, labels)
    overall = jnp.mean(values)
    return jnp.sum(counts * (means - overall) ** 2)


def _labels(n: int, n1: int) -> jax.Array:
    return jnp.concatenate([jnp.zeros(n - n1, jnp.int32), jnp.ones(n1, jnp.int32)])


def random_population(key: jax.Array, n: int, n1: int, scale=(1.0, 1.0), shift=(0.0, 0.0)) -> Population:
    """A fold with Gaussian unweighted losses, class c scaled by ``scale[c]``.

    Nothing here is meant to imitate a real loss surface. It gives a population with a
    known N_0, N_1 and dispersions that differ between classes, which is all the
    design-based results need.
    """
    labels = _labels(n, n1)
    z = jax.random.normal(key, (n,), dtype=jnp.float64)
    sc = jnp.where(labels == 1, scale[1], scale[0])
    sh = jnp.where(labels == 1, shift[1], shift[0])
    raw = sh + sc * z
    return Population(losses=weighted_losses(raw, labels), labels=labels, raw=raw)


def equidispersed_population(key: jax.Array, n: int, n1: int, sigma: float = 1.0) -> Population:
    """A fold satisfying condition (13) of Corollary 9, sigma_1 = sigma_0, exactly.

    Each class's unweighted losses are recentred and rescaled so that its sample
    dispersion (7) equals ``sigma`` to floating point, which is what makes equal
    allocation exactly Neyman-optimal and puts Corollary 10's factor 1/{4 pi (1 - pi)}
    on the nose rather than approximately.
    """
    labels = _labels(n, n1)
    z = jax.random.normal(key, (n,), dtype=jnp.float64)
    means = class_means(z, labels)
    disp = class_dispersions(z, labels)
    per_row_mean = jnp.where(labels == 1, means[1], means[0])
    per_row_disp = jnp.where(labels == 1, disp[1], disp[0])
    raw = sigma * (z - per_row_mean) / per_row_disp
    # Give the classes different means so the between-class term B of Proposition 6 is
    # not degenerate; a shift does not touch S_c.
    raw = raw + jnp.where(labels == 1, 3.0 * sigma, 0.0)
    return Population(losses=weighted_losses(raw, labels), labels=labels, raw=raw)
