"""The sampler for pi_xi, Section 3.3.

The reference distribution is never the solver's own tabulated cdf, since that
would only say the sampler inverts what it was handed.  For constant xi the
energy law is a Gamma in closed form; for the other coefficients the reference
cdf is built by a separate quadrature of f_xi(s) = q_xi(s) m_V(s) in the energy
variable, on a different grid from the radial one the solver uses.

Thresholds come from the Kolmogorov distribution, not from looking at a seed.
"""

import math

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import gammainc

from mkv_sieve import (
    OUPotential,
    profile_energy_cdf,
    profile_log_q,
    sample_energies,
    sample_particles,
    solve_profile,
)
from mkv_sieve.quad import cum_quad4
from mkv_sieve.truth import ENVELOPE, TRUTH

N = 20_000
SEEDS = range(5)


def ks_pvalue(dn, n):
    """Asymptotic Kolmogorov tail, with Stephens' small-sample correction."""
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * dn
    k = jnp.arange(1, 101)
    return float(2.0 * jnp.sum((-1.0) ** (k - 1) * jnp.exp(-2.0 * k * k * lam * lam)))


def ks_statistic(sample, cdf):
    n = sample.shape[0]
    s = jnp.sort(sample)
    f = cdf(s)
    k = jnp.arange(1, n + 1)
    return float(jnp.maximum(jnp.max(k / n - f), jnp.max(f - (k - 1) / n)))


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("c", [0.9, 1.3])
def test_energies_follow_the_closed_form_gamma_law(d, c):
    """For xi == c, S - v_star is Gamma(d/2, rate c).

    q_xi(s) m_V(s) = (c^{d/2} / Gamma(d/2)) (s-1)^{d/2-1} exp(-c(s-1)) once
    a_xi = (c/pi)^{d/2} is put into f_xi, so the energy law is exactly known.
    """
    pot = OUPotential(d)
    prof = solve_profile(
        lambda r: jnp.full_like(jnp.asarray(r), c), pot, 0.8 * c, 1.3 * c
    )
    for seed in SEEDS:
        s = sample_energies(jax.random.PRNGKey(seed), prof, N)
        dn = ks_statistic(s, lambda t: gammainc(d / 2.0, c * (t - 1.0)))
        assert ks_pvalue(dn, N) > 0.01


@pytest.mark.parametrize("d", [2, 4])
@pytest.mark.parametrize("which", [2, 3])
def test_energies_follow_f_xi_for_nonconstant_xi(d, which):
    """S has density f_xi(s) = q_xi(s) m_V(s), Section 3.3.

    The reference cdf here is a cumulative quadrature in s against m_V from
    Remark 3.2, which is a different construction from the radial one the
    sampler inverts.  d is even so m_V has no singularity at v_star.
    """
    kappa, K = ENVELOPE[which]
    pot = OUPotential(d)
    prof = solve_profile(TRUTH[which], pot, kappa, K)

    s_max = 1.0 + float(prof.rho[-1]) ** 2
    grid = jnp.linspace(1.0, s_max, 20001)
    f = jnp.exp(profile_log_q(prof, grid)) * pot.m_V(grid)
    cdf = cum_quad4(f, grid[1] - grid[0])
    assert float(jnp.abs(cdf[-1] - 1.0)) < 1e-8

    for seed in SEEDS:
        s = sample_energies(jax.random.PRNGKey(seed), prof, N)
        dn = ks_statistic(s, lambda t: jnp.interp(t, grid, cdf))
        assert ks_pvalue(dn, N) > 0.01


@pytest.mark.parametrize("which", [2, 3])
def test_tabulated_cdf_agrees_with_the_direct_energy_quadrature(which):
    """profile_energy_cdf, built radially, matches the s-variable quadrature."""
    kappa, K = ENVELOPE[which]
    pot = OUPotential(2)
    prof = solve_profile(TRUTH[which], pot, kappa, K)
    s_max = 1.0 + float(prof.rho[-1]) ** 2
    grid = jnp.linspace(1.0, s_max, 20001)
    f = jnp.exp(profile_log_q(prof, grid)) * pot.m_V(grid)
    cdf = cum_quad4(f, grid[1] - grid[0])
    query = jnp.linspace(1.0, s_max, 97)
    diff = jnp.abs(jnp.interp(query, grid, cdf) - profile_energy_cdf(prof, query))
    assert float(jnp.max(diff)) < 1e-6


@pytest.mark.parametrize("d", [2, 3])
def test_particles_carry_the_right_energy_law_and_are_isotropic(d):
    """X in R^d from Section 3.3: draw S, then a uniform point on {V = S}."""
    kappa, K = ENVELOPE[3]
    pot = OUPotential(d)
    prof = solve_profile(TRUTH[3], pot, kappa, K)
    x = sample_particles(jax.random.PRNGKey(7), prof, pot, N)
    assert x.shape == (N, d)

    dn = ks_statistic(pot.V(x), lambda t: profile_energy_cdf(prof, t))
    assert ks_pvalue(dn, N) > 0.01

    # directions uniform on the sphere: the mean is 0 to Monte Carlo accuracy
    u = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    assert float(jnp.max(jnp.abs(jnp.mean(u, axis=0)))) < 4.0 / math.sqrt(N)
