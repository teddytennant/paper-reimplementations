"""Properties of the GHST density displayed in Section 2.4 and its stated limits."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

from conftest import gauss_legendre_panels
from dstmm import ghst_cov, ghst_logpdf, ghst_mean, sample_ghst, student_t_logpdf

MU1 = jnp.array([0.3])
SIG1 = jnp.array([[1.4]])
AL1 = jnp.array([0.7])

MU2 = jnp.array([0.3, -0.2])
SIG2 = jnp.array([[1.4, 0.3], [0.3, 0.9]])
AL2 = jnp.array([0.7, -0.4])


def test_density_integrates_to_one_p1():
    """The Section 2.4 density is a density. Composite Gauss-Legendre in one dimension.

    The box is asymmetric because a GHST has a polynomial tail on the alpha side
    and a lighter one on the other; at nu = 8 the mass outside it is below 1e-9.
    """
    nodes, weights = gauss_legendre_panels(-80.0, 200.0, 600)
    vals = np.asarray(jnp.exp(ghst_logpdf(jnp.asarray(nodes)[:, None], MU1, SIG1, AL1, 8.0)))
    assert abs(float((vals * weights).sum()) - 1.0) < 1e-7


def test_density_integrates_to_one_p2():
    """Same statement in two dimensions on a tensor-product rule."""
    nodes, weights = gauss_legendre_panels(-40.0, 80.0, 35)
    grid = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)
    wts = (weights[:, None] * weights[None, :]).ravel()
    vals = np.asarray(jnp.exp(ghst_logpdf(jnp.asarray(grid), MU2, SIG2, AL2, 8.0)))
    assert abs(float((vals * wts).sum()) - 1.0) < 5e-6


def test_zero_skewness_is_the_multivariate_t():
    """Corollary 3 part 1 at the density level: alpha = 0 gives the Student-t exactly."""
    y = jax.random.normal(jax.random.key(0), (60, 2), dtype=jnp.float64) * 2.0
    nu = 6.0
    got = np.asarray(ghst_logpdf(y, MU2, SIG2, jnp.zeros(2), nu))
    mine = np.asarray(student_t_logpdf(y, MU2, SIG2, nu))
    ref = multivariate_t.logpdf(np.asarray(y), loc=np.asarray(MU2), shape=np.asarray(SIG2), df=nu)
    assert np.abs(got - mine).max() == 0.0
    assert np.abs(got - ref).max() < 1e-13


def test_bessel_branch_converges_to_the_t_at_first_order():
    """Section 2.4 calls the Student-t the continuous alpha -> 0 limit.

    Checked through the Bessel branch itself, not the guarded switch: the gap has
    to shrink like ||alpha||, so dividing alpha by 10 divides the gap by 10.
    """
    y = jax.random.normal(jax.random.key(1), (40, 2), dtype=jnp.float64) * 2.0
    nu = 6.0
    ref = multivariate_t.logpdf(np.asarray(y), loc=np.asarray(MU2), shape=np.asarray(SIG2), df=nu)
    gaps = []
    for eps in (1e-3, 1e-4, 1e-5):
        got = np.asarray(ghst_logpdf(y, MU2, SIG2, jnp.array([eps, 0.0]), nu))
        gaps.append(np.abs(got - ref).max())
    assert gaps[0] < 1e-2
    for a, b in zip(gaps[:-1], gaps[1:]):
        assert 9.0 < a / b < 11.0


def test_gaussian_limit():
    """Corollary 3 part 2: as nu -> infinity the component tends to N(mu + alpha, Sigma),
    and the gap has to close like 1/nu."""
    y = jax.random.normal(jax.random.key(2), (40, 2), dtype=jnp.float64) * 2.0
    ref = multivariate_normal.logpdf(
        np.asarray(y), mean=np.asarray(MU2 + AL2), cov=np.asarray(SIG2)
    )
    gaps = []
    for nu in (1e4, 1e5, 1e6):
        gaps.append(np.abs(np.asarray(ghst_logpdf(y, MU2, SIG2, AL2, nu)) - ref).max())
    assert gaps[-1] < 1e-3
    for a, b in zip(gaps[:-1], gaps[1:]):
        assert 9.0 < a / b < 11.0


def test_moments_of_the_representation():
    """E(X) = mu + E(W) alpha and Cov(X) = E(W) Sigma + Var(W) alpha alpha' (Section 2.1),
    checked against draws from the mean-variance representation (1)-(2)."""
    nu = 12.0  # above 8, so the sample covariance itself has finite variance
    n = 200_000
    y, _ = sample_ghst(jax.random.key(3), MU2, SIG2, AL2, nu, n)
    y = np.asarray(y)

    mean_ref = np.asarray(ghst_mean(MU2, AL2, nu))
    se_mean = y.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(y.mean(axis=0) - mean_ref) < 5.0 * se_mean)

    cov_ref = np.asarray(ghst_cov(SIG2, AL2, nu))
    c = y - y.mean(axis=0)
    prod = c[:, :, None] * c[:, None, :]
    se_cov = prod.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(prod.mean(axis=0) - cov_ref) < 5.0 * se_cov)
