"""The two algorithms the paper measures itself against.

``kmeans`` is Lloyd's classical Euclidean K-means. ``mkmeans`` is Algorithm 1
of arXiv:2608.30093v1, K-means under the Mahalanobis distance with maximum
likelihood updates for (mu_k, Sigma_k). Both are here so the robustness claim
of Section 4.2 can actually be tested rather than asserted.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dpd import _PREC, default_cov_floor, mahalanobis, spec_from_cov


class KMeansResult(NamedTuple):
    centers: jnp.ndarray  # (K, p)
    labels: jnp.ndarray  # (n,)
    n_iter: int


class MKMeansResult(NamedTuple):
    centers: jnp.ndarray  # (K, p)
    covariances: jnp.ndarray  # (K, p, p)
    labels: jnp.ndarray  # (n,)
    n_iter: int


def kmeans_plusplus(x: jnp.ndarray, k: int, key) -> jnp.ndarray:
    """k-means++ seeding (Arthur and Vassilvitskii). Returns (K, p) centers.

    The paper uses the core-neighbourhood scheme of its Section 3.2.1 instead;
    that one lives in ``init.core_init``. This is the cheap default.
    """
    keys = jax.random.split(key, k)
    first = jax.random.randint(keys[0], (), 0, x.shape[0])
    centers = [x[first]]
    d2 = jnp.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / jnp.maximum(jnp.sum(d2), 1e-12)
        idx = jax.random.choice(keys[j], x.shape[0], p=probs)
        centers.append(x[idx])
        d2 = jnp.minimum(d2, jnp.sum((x - centers[-1]) ** 2, axis=1))
    return jnp.stack(centers)


def _euclidean_assign(x: jnp.ndarray, centers: jnp.ndarray) -> jnp.ndarray:
    d = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(d, axis=1)


def _one_hot(labels: jnp.ndarray, k: int) -> jnp.ndarray:
    return jax.nn.one_hot(labels, k, dtype=jnp.result_type(float))


def _means(x: jnp.ndarray, resp: jnp.ndarray, fallback: jnp.ndarray) -> jnp.ndarray:
    nk = jnp.sum(resp, axis=0)
    safe = jnp.where(nk > 0, nk, 1.0)
    m = jnp.einsum("nk,nj->kj", resp, x, precision=_PREC) / safe[:, None]
    return jnp.where(nk[:, None] > 0, m, fallback)


def _ml_covariances(
    x: jnp.ndarray, resp: jnp.ndarray, mu: jnp.ndarray, fallback: jnp.ndarray
) -> jnp.ndarray:
    """MLE covariance of Step 5 of Algorithm 1: (1/n_k) sum (x - mu)(x - mu)'."""
    nk = jnp.sum(resp, axis=0)
    safe = jnp.where(nk > 0, nk, 1.0)
    dx = x[:, None, :] - mu[None, :, :]
    s = jnp.einsum("nk,nki,nkj->kij", resp, dx, dx, precision=_PREC)
    s = s / safe[:, None, None]
    return jnp.where(nk[:, None, None] > 0, s, fallback)


def kmeans(x, k, key=None, centers=None, max_iter=200) -> KMeansResult:
    """Classical Euclidean K-means, minimizing Delta(C_1, ..., C_K) of Section 2."""
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    if centers is None:
        centers = kmeans_plusplus(x, k, key)
    centers = jnp.asarray(centers, dtype=x.dtype)
    labels = _euclidean_assign(x, centers)
    n_iter = 0
    for it in range(max_iter):
        resp = _one_hot(labels, k)
        centers = _means(x, resp, centers)
        new = _euclidean_assign(x, centers)
        n_iter = it + 1
        if bool(jnp.all(new == labels)):
            labels = new
            break
        labels = new
    return KMeansResult(centers, labels, n_iter)


def mkmeans(
    x, k, key=None, centers=None, max_iter=200, cov_floor=None
) -> MKMeansResult:
    """Algorithm 1, K-means using Mahalanobis distance (MK-means).

    Steps 1 and 2 seed with Euclidean distance, Step 3 takes the MLE covariance
    of that seed partition, then Steps 4 and 5 alternate Mahalanobis assignment
    with MLE re-estimation. Section 3.3 spells out why this loop has no
    convergence guarantee: the assignment objective D_Mahal has no dependence on
    Lambda_k, so it is not the objective the update step minimizes. We stop on
    a repeated partition or on ``max_iter``, whichever comes first, and make no
    claim that the distortion fell.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    if cov_floor is None:
        cov_floor = default_cov_floor(x)
    if centers is None:
        centers = kmeans_plusplus(x, k, key)
    mu = jnp.asarray(centers, dtype=x.dtype)

    labels = _euclidean_assign(x, mu)  # Step 2
    resp = _one_hot(labels, k)
    eye = jnp.broadcast_to(
        jnp.eye(x.shape[1], dtype=x.dtype), (k, x.shape[1], x.shape[1])
    )
    sigma = _ml_covariances(x, resp, mu, eye)  # Step 3

    n_iter = 0
    for it in range(max_iter):
        spec = spec_from_cov(sigma, cov_floor)
        new = jnp.argmin(mahalanobis(x, mu, spec), axis=1)  # Step 4
        n_iter = it + 1
        done = bool(jnp.all(new == labels))
        labels = new
        resp = _one_hot(labels, k)
        mu = _means(x, resp, mu)  # Step 5
        sigma = _ml_covariances(x, resp, mu, sigma)
        if done:
            break
    spec = spec_from_cov(sigma, cov_floor)
    return MKMeansResult(mu, spec.sigma, labels, n_iter)
