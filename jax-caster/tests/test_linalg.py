"""The symmetric matrix functions eq. (3) and eq. (9) are built on."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from caster.linalg import (  # noqa: E402
    mahalanobis_sq,
    psd_inv_sqrt,
    psd_logdet_and_quadform,
    psd_sqrt,
)
from caster.synthetic import random_spd  # noqa: E402


@pytest.mark.parametrize("condition", [2.0, 1e4])
def test_psd_sqrt_squares_back(condition):
    m = random_spd(jax.random.PRNGKey(0), 8, condition=condition)
    root = psd_sqrt(m)
    assert np.max(np.abs(np.asarray(root - root.T))) < 1e-14
    scale = float(np.max(np.abs(np.asarray(m))))
    assert np.max(np.abs(np.asarray(root @ root - m))) < 1e-12 * scale


@pytest.mark.parametrize("condition", [2.0, 1e4])
def test_inv_sqrt_is_the_inverse_of_sqrt(condition):
    m = random_spd(jax.random.PRNGKey(1), 8, condition=condition)
    product = psd_inv_sqrt(m) @ psd_sqrt(m)
    assert np.max(np.abs(np.asarray(product - jnp.eye(8)))) < 1e-10


def test_sqrt_clamps_a_semidefinite_matrix_instead_of_returning_nan():
    """A rank-deficient covariance has a real root; a naive solve would not."""
    key = jax.random.PRNGKey(2)
    factor = jax.random.normal(key, (6, 3), dtype=jnp.float64)
    m = factor @ factor.T  # rank 3 in 6 dimensions
    root = psd_sqrt(m)
    assert np.all(np.isfinite(np.asarray(root)))
    assert np.max(np.abs(np.asarray(root @ root - m))) < 1e-11 * float(np.max(np.abs(np.asarray(m))))


def test_logdet_and_quadform_match_numpy():
    key_m, key_r = jax.random.split(jax.random.PRNGKey(3))
    m = random_spd(key_m, 5, condition=50.0)
    residuals = jax.random.normal(key_r, (20, 5), dtype=jnp.float64)
    logdet, quad = psd_logdet_and_quadform(m[None], residuals[:, None, :])

    inverse = np.linalg.inv(np.asarray(m))
    expected = np.einsum("ni,ij,nj->n", np.asarray(residuals), inverse, np.asarray(residuals))
    assert abs(float(logdet[0]) - float(np.linalg.slogdet(np.asarray(m))[1])) < 1e-11
    assert np.max(np.abs(np.asarray(quad)[:, 0] - expected)) < 1e-9


def test_mahalanobis_matches_an_explicit_inverse():
    key_m, key_v = jax.random.split(jax.random.PRNGKey(4))
    m = random_spd(key_m, 6, condition=100.0)
    vectors = jax.random.normal(key_v, (15, 6), dtype=jnp.float64)
    inverse = np.linalg.inv(np.asarray(m))
    expected = np.einsum("ni,ij,nj->n", np.asarray(vectors), inverse, np.asarray(vectors))
    assert np.max(np.abs(np.asarray(mahalanobis_sq(m, vectors)) - expected)) < 1e-9
