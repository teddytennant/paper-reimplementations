"""Equations (7) to (9) and the accept rule of Sec. 3.2."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from caster import Caster, CasterConfig  # noqa: E402
from caster.certificate import (  # noqa: E402
    Certificate,
    centroid_residuals,
    gate,
    mixed_metric_certificate,
    residual_to_margin,
)
from caster.source import min_class_separation  # noqa: E402
from caster.synthetic import class_parameters, random_spd, sample_classes  # noqa: E402


def _random_certificate_inputs(seed, num_classes=6, k=5):
    key = jax.random.PRNGKey(seed)
    key_mu, key_cov, key_res = jax.random.split(key, 3)
    means = jax.random.normal(key_mu, (num_classes, k), dtype=jnp.float64) * 3.0
    pooled = random_spd(key_cov, k, condition=25.0)
    residuals = jax.random.normal(key_res, (num_classes, k), dtype=jnp.float64) * 0.4
    occupied = np.ones(num_classes, dtype=bool)
    return means, pooled, residuals, occupied


def test_gamma_s_is_the_minimum_pairwise_mahalanobis_gap():
    """Eq. (8), checked against an explicit double loop."""
    means, pooled, _, _ = _random_certificate_inputs(0)
    inverse = np.linalg.inv(np.asarray(pooled))
    gaps = []
    for i in range(means.shape[0]):
        for j in range(means.shape[0]):
            if i == j:
                continue
            diff = np.asarray(means[i] - means[j])
            gaps.append(np.sqrt(diff @ inverse @ diff))
    assert abs(float(min_class_separation(means, pooled)) - min(gaps)) < 1e-10


def test_certificate_is_invariant_to_an_invertible_reparametrization():
    """Eq. (9) does not move under z -> Lz, and the mixed-metric version does.

    Sec. 3.2's argument: the residual and the margin are both Mahalanobis
    lengths in Sigma_s, so under z -> Lz they pick up the same factor and the
    ratio is unchanged. Swap the numerator for a Euclidean norm and that
    cancellation is gone. L below stretches one feature coordinate by 50 and
    squashes another by 20, which is mild next to the conditioning spread of a
    real backbone, and it already moves the mixed-metric score by more than an
    order of magnitude. No fixed threshold on that score means the same thing
    in two coordinate systems.
    """
    means, pooled, residuals, occupied = _random_certificate_inputs(1)
    k = means.shape[1]
    scales = jnp.asarray([50.0, 1.0, 0.05, 1.0, 2.0][:k], dtype=jnp.float64)
    lmap = jnp.diag(scales)

    means_l = means @ lmap.T
    pooled_l = lmap @ pooled @ lmap.T
    residuals_l = residuals @ lmap.T

    gamma = min_class_separation(means, pooled)
    gamma_l = min_class_separation(means_l, pooled_l)
    assert abs(float(gamma_l) - float(gamma)) < 1e-8 * float(gamma)

    base, _ = residual_to_margin(residuals, occupied, pooled, gamma, 0.0)
    mapped, _ = residual_to_margin(residuals_l, occupied, pooled_l, gamma_l, 0.0)
    assert abs(float(mapped) - float(base)) < 1e-8 * float(base)

    mixed = float(mixed_metric_certificate(residuals, occupied, pooled, gamma, 0.0))
    mixed_l = float(mixed_metric_certificate(residuals_l, occupied, pooled_l, gamma_l, 0.0))
    assert max(mixed_l / mixed, mixed / mixed_l) > 10.0


def test_centroid_residuals_follow_eq7():
    """delta_c is the mean of the inverse-transported retained samples minus mu_c."""
    rng = np.random.default_rng(0)
    num_classes, k, n = 4, 3, 50
    means = rng.normal(size=(num_classes, k))
    zbar = rng.normal(size=(n, k))
    pseudo = rng.integers(0, num_classes - 1, size=n)  # class 3 stays empty

    residuals, occupied = centroid_residuals(zbar, pseudo, means, num_classes)
    assert occupied.tolist() == [True, True, True, False]
    for c in range(num_classes - 1):
        expected = zbar[pseudo == c].mean(axis=0) - means[c]
        assert np.max(np.abs(residuals[c] - expected)) < 1e-12
    assert np.all(residuals[3] == 0.0)


def test_unoccupied_classes_do_not_enter_the_max_in_eq9():
    """The max in eq. (9) runs over c with |I_c| > 0 only."""
    means, pooled, residuals, _ = _random_certificate_inputs(4)
    gamma = min_class_separation(means, pooled)
    huge = residuals.at[0].set(residuals[0] * 500.0)
    occupied = np.ones(means.shape[0], dtype=bool)
    occupied[0] = False
    with_huge, _ = residual_to_margin(huge, occupied, pooled, gamma, 0.0)
    without, _ = residual_to_margin(residuals, occupied, pooled, gamma, 0.0)
    assert abs(float(with_huge) - float(without)) < 1e-12


def test_gate_enforces_tau_and_both_coverage_requirements():
    """Sec. 3.2: r_t <= tau, enough retained samples, enough occupied classes."""
    kwargs = dict(num_classes=20, tau=0.8, min_retained=8, min_class_coverage=0.25)
    passing = Certificate(value=0.5, n_retained=40, n_occupied=6, gamma_s=1.0, residual=0.5)
    assert gate(passing, **kwargs).accepted

    assert not gate(passing._replace(value=0.81), **kwargs).accepted
    assert "tau" in gate(passing._replace(value=0.81), **kwargs).reason

    thin = gate(passing._replace(n_retained=7), **kwargs)
    assert not thin.accepted and "samples" in thin.reason

    narrow = gate(passing._replace(n_occupied=4), **kwargs)  # needs ceil(0.25 * 20) = 5
    assert not narrow.accepted and "classes" in narrow.reason

    empty = gate(passing._replace(n_occupied=0), **kwargs)
    assert not empty.accepted and "no confident" in empty.reason


def test_pipeline_certificate_survives_rescaling_the_raw_features():
    """End to end: rescale h, refit, re-adapt, r_t and the predictions hold.

    This is the composition of two facts, that the LDA projection of eq. (1) is
    contravariant in the features and that eq. (9) is a ratio of two lengths in
    the same metric. Nothing here is exact, since the within-class ridge does
    not commute with L, so the tolerance is 1e-6 relative rather than 1e-12.
    """
    num_classes, dim = 12, 16
    params = class_parameters(jax.random.PRNGKey(5), num_classes, dim, separation=2.5)
    source, source_labels = sample_classes(jax.random.PRNGKey(6), params, 200)
    target, _ = sample_classes(jax.random.PRNGKey(7), params, 20)

    scales = jnp.exp(
        jax.random.uniform(jax.random.PRNGKey(8), (dim,), dtype=jnp.float64) * 2.0 - 1.0
    )
    rotation, _ = jnp.linalg.qr(
        jax.random.normal(jax.random.PRNGKey(9), (dim, dim), dtype=jnp.float64)
    )
    lmap = rotation * scales[None, :]

    config = CasterConfig(within_ridge=1e-12)
    plain = Caster.fit(source, source_labels, num_classes, config)
    rescaled = Caster.fit(source @ lmap.T, source_labels, num_classes, config)

    a = plain.adapt(target)
    b = rescaled.adapt(target @ lmap.T)
    assert abs(b.certificate - a.certificate) < 1e-6 * a.certificate
    assert np.array_equal(np.asarray(a.predictions), np.asarray(b.predictions))
