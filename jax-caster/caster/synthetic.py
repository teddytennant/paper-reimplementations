"""Synthetic Gaussian features, standing in for a frozen backbone.

The paper extracts features from ViT-B/16, ResNet-50, DeiT-B and Swin-T. This
package takes feature vectors and labels, so the demo and the tests generate
class-conditional Gaussians directly. Nothing in `caster` depends on this
module.

`class_parameters` and `sample_classes` are split so a target set can be drawn
from the same class geometry as the source set and then shifted. Drawing both
from one call with different keys gives two unrelated problems, not a shift.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ClassParameters(NamedTuple):
    """Per-class means and Cholesky factors of the covariances."""

    means: jnp.ndarray   # (C, d)
    roots: jnp.ndarray   # (C, d, d)


def random_spd(key, dim, condition=10.0, dtype=jnp.float64):
    """A random symmetric positive definite matrix with a set condition number."""
    key_q, key_w = jax.random.split(key)
    q, _ = jnp.linalg.qr(jax.random.normal(key_q, (dim, dim), dtype=dtype))
    logs = jax.random.uniform(key_w, (dim,), dtype=dtype)
    eigenvalues = jnp.exp(logs * jnp.log(jnp.asarray(condition, dtype)))
    return (q * eigenvalues[None, :]) @ q.T


def class_parameters(
    key,
    num_classes,
    dim,
    separation=3.0,
    condition=6.0,
    shared_covariance=True,
    dtype=jnp.float64,
):
    """Draw the class geometry: C means and C covariance factors.

    With `shared_covariance` the classes differ only in their means, which is
    the regime LDA and the Gaussian discriminant of eq. (5) are built for.
    """
    key_mu, key_cov = jax.random.split(key)
    means = jax.random.normal(key_mu, (num_classes, dim), dtype=dtype) * separation
    if shared_covariance:
        root = jnp.linalg.cholesky(random_spd(key_cov, dim, condition, dtype))
        roots = jnp.broadcast_to(root, (num_classes, dim, dim))
    else:
        keys = jax.random.split(key_cov, num_classes)
        roots = jnp.stack([
            jnp.linalg.cholesky(random_spd(k, dim, condition, dtype)) for k in keys
        ])
    return ClassParameters(means=means, roots=roots)


def sample_classes(key, params: ClassParameters, n_per_class):
    """Draw n_per_class samples from every class in `params`."""
    num_classes, dim = params.means.shape
    noise = jax.random.normal(
        key, (num_classes, n_per_class, dim), dtype=params.means.dtype
    )
    samples = params.means[:, None, :] + jnp.einsum("cij,cnj->cni", params.roots, noise)
    labels = jnp.repeat(jnp.arange(num_classes), n_per_class)
    return samples.reshape(num_classes * n_per_class, dim), labels


def gaussian_classes(key, num_classes, dim, n_per_class, **kwargs):
    """Draw parameters and one sample set in a single call."""
    key_params, key_draw = jax.random.split(key)
    params = class_parameters(key_params, num_classes, dim, **kwargs)
    features, labels = sample_classes(key_draw, params, n_per_class)
    return features, labels, params


def affine_shift(features, matrix, offset):
    """t = A s + b, the class-shared shift CASTER assumes in Sec. 3.1."""
    return jnp.asarray(features) @ jnp.asarray(matrix).T + jnp.asarray(offset)[None, :]
