"""Equations (3), (4) and (6)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from caster.linalg import psd_sqrt, ridge  # noqa: E402
from caster.synthetic import random_spd  # noqa: E402
from caster.transport import (  # noqa: E402
    empirical_moments,
    estimate_transport,
    inverse_transport,
    transport_class_moments,
    use_diagonal_form,
    whiten_color_map,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_whiten_color_map_recovers_a_known_spd_shift(seed):
    """Eq. (3) returns A_true and b_true exactly when C_s = I and A_true is SPD.

    A_t = (Chat_t + lam I)^{1/2} (C_s + lam I)^{-1/2}. With C_s = I and
    Chat_t = A C_s A^T = A^2, the first factor is A itself in the lam -> 0
    limit, so the map is identifiable. For a general C_s it is not: any
    A' = Chat^{1/2} R C_s^{-1/2} with R orthogonal reproduces the same second
    moments, and eq. (3) picks R = I in the whitened frame, not R = the one
    that happens to be A_true. `test_map_carries_source_covariance_onto_target`
    below is the property that does hold for every C_s.
    """
    key = jax.random.PRNGKey(seed)
    k = 6
    key_a, key_ms, key_b = jax.random.split(key, 3)
    a_true = psd_sqrt(random_spd(key_a, k, condition=9.0))  # SPD, spread ~1 to 3
    m_s = jax.random.normal(key_ms, (k,), dtype=jnp.float64) * 2.0
    b_true = jax.random.normal(key_b, (k,), dtype=jnp.float64) * 5.0

    c_s = jnp.eye(k, dtype=jnp.float64)
    c_hat = a_true @ c_s @ a_true.T
    m_hat = a_true @ m_s + b_true

    # The error here is the lambda bias, not roundoff: it scales linearly with
    # lam, so 1e-12 puts it at the float64 floor. At the default lam = 1e-3 the
    # same test passes only to about 1e-3.
    affine = whiten_color_map(m_s, c_s, m_hat, c_hat, lam=1e-12, diagonal=False)
    assert np.max(np.abs(np.asarray(affine.a) - np.asarray(a_true))) < 1e-10
    assert np.max(np.abs(np.asarray(affine.b) - np.asarray(b_true))) < 1e-10


def test_estimate_transport_recovers_a_known_shift_from_samples():
    """Same recovery, but going through the batch moments of a real sample.

    The source sample is standardized so its empirical mean is 0 and its
    empirical covariance is exactly I, which is the C_s = I condition of the
    test above without appealing to a large-sample argument.
    """
    key = jax.random.PRNGKey(1)
    k, n = 5, 400
    key_x, key_a, key_b = jax.random.split(key, 3)
    raw = jax.random.normal(key_x, (n, k), dtype=jnp.float64)
    mean, cov = empirical_moments(raw)
    whitened = (raw - mean) @ np.linalg.inv(np.linalg.cholesky(np.asarray(cov))).T

    m_s = jnp.zeros((k,), dtype=jnp.float64)
    a_true = psd_sqrt(random_spd(key_a, k, condition=16.0))
    b_true = jax.random.normal(key_b, (k,), dtype=jnp.float64) * 3.0
    target = whitened @ a_true.T + b_true[None, :]

    affine = estimate_transport(m_s, jnp.eye(k), target, lam=1e-12)
    assert not affine.diagonal
    assert np.max(np.abs(np.asarray(affine.a) - np.asarray(a_true))) < 1e-8
    assert np.max(np.abs(np.asarray(affine.b) - np.asarray(b_true))) < 1e-8


def test_map_carries_source_covariance_onto_target_covariance():
    """A_t (C_s + lam I) A_t^T = Chat_t + lam I, for any C_s and any lam.

    This is the defining property of eq. (3) and it does not need C_s = I.
    """
    key = jax.random.PRNGKey(2)
    k = 7
    key_s, key_t, key_m = jax.random.split(key, 3)
    c_s = random_spd(key_s, k, condition=50.0)
    c_hat = random_spd(key_t, k, condition=50.0)
    m_hat = jax.random.normal(key_m, (k,), dtype=jnp.float64)
    lam = 1e-3

    affine = whiten_color_map(jnp.zeros(k), c_s, m_hat, c_hat, lam=lam, diagonal=False)
    lhs = affine.a @ ridge(c_s, lam) @ affine.a.T
    rhs = ridge(c_hat, lam)
    assert np.max(np.abs(np.asarray(lhs - rhs))) < 1e-10 * float(np.max(np.abs(np.asarray(rhs))))


@pytest.mark.parametrize("k", [3, 8])
def test_diagonal_form_triggers_exactly_below_two_k(k):
    """Sec. 3.1's B < 2k rule, checked on both sides of the boundary."""
    assert use_diagonal_form(2 * k - 1, k) is True
    assert use_diagonal_form(2 * k, k) is False

    key = jax.random.PRNGKey(3)
    c_s = random_spd(key, k, condition=4.0)
    for batch_size, expected in [(2 * k - 1, True), (2 * k, False)]:
        batch = jax.random.normal(
            jax.random.PRNGKey(batch_size), (batch_size, k), dtype=jnp.float64
        )
        affine = estimate_transport(jnp.zeros(k), c_s, batch, lam=1e-3)
        assert affine.diagonal is expected
        assert affine.a.shape == (k, k)


def test_diagonal_and_full_forms_agree_when_covariances_are_diagonal():
    """The two branches of eq. (3) coincide when the covariances really are diagonal."""
    key = jax.random.PRNGKey(4)
    k = 6
    key_s, key_t, key_m = jax.random.split(key, 3)
    c_s = jnp.diag(jax.random.uniform(key_s, (k,), dtype=jnp.float64, minval=0.5, maxval=4.0))
    c_hat = jnp.diag(jax.random.uniform(key_t, (k,), dtype=jnp.float64, minval=0.5, maxval=4.0))
    m_s = jnp.zeros(k)
    m_hat = jax.random.normal(key_m, (k,), dtype=jnp.float64)

    full = whiten_color_map(m_s, c_s, m_hat, c_hat, lam=1e-3, diagonal=False)
    diag = whiten_color_map(m_s, c_s, m_hat, c_hat, lam=1e-3, diagonal=True)
    assert np.max(np.abs(np.asarray(full.a - diag.a))) < 1e-12
    assert np.max(np.abs(np.asarray(full.b - diag.b))) < 1e-12


def test_transported_moments_follow_eq4():
    """mu~ = A mu + b and Sigma~ = A Sigma A^T + beta I, term by term."""
    key = jax.random.PRNGKey(5)
    k, c = 4, 5
    key_a, key_b, key_mu, key_cov = jax.random.split(key, 4)
    affine = whiten_color_map(
        jnp.zeros(k),
        random_spd(key_a, k, condition=6.0),
        jax.random.normal(key_b, (k,), dtype=jnp.float64),
        random_spd(key_a, k, condition=3.0),
        lam=1e-3,
        diagonal=False,
    )
    means = jax.random.normal(key_mu, (c, k), dtype=jnp.float64)
    covs = jnp.stack([random_spd(k_, k, condition=5.0) for k_ in jax.random.split(key_cov, c)])
    beta = 1e-2

    t_means, t_covs = transport_class_moments(affine, means, covs, beta)
    for i in range(c):
        expected_mean = affine.a @ means[i] + affine.b
        expected_cov = affine.a @ covs[i] @ affine.a.T + beta * jnp.eye(k)
        assert np.allclose(np.asarray(t_means[i]), np.asarray(expected_mean), atol=1e-12)
        assert np.allclose(np.asarray(t_covs[i]), np.asarray(expected_cov), atol=1e-12)


def test_inverse_transport_undoes_eq3():
    """Eq. (6): zbar = A^-1 (z - b) is the exact inverse of z = A zbar + b."""
    key = jax.random.PRNGKey(6)
    k, n = 5, 40
    key_a, key_b, key_z = jax.random.split(key, 3)
    affine = whiten_color_map(
        jnp.zeros(k),
        random_spd(key_a, k, condition=8.0),
        jax.random.normal(key_b, (k,), dtype=jnp.float64),
        random_spd(key_b, k, condition=8.0),
        lam=1e-4,
        diagonal=False,
    )
    zbar = jax.random.normal(key_z, (n, k), dtype=jnp.float64)
    z = zbar @ affine.a.T + affine.b[None, :]
    assert np.max(np.abs(np.asarray(inverse_transport(affine, z) - zbar))) < 1e-10
