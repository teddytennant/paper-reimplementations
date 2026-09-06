"""The forward map of Proposition 3.4, checked against things the paper states exactly."""

import math

import jax
import jax.numpy as jnp
import pytest

from mkv_sieve import (
    OUPotential,
    profile_a,
    profile_log_q,
    solve_profile,
)
from mkv_sieve.profile import profile_mass, profile_pi
from mkv_sieve.quad import simpson_weights
from mkv_sieve.truth import ENVELOPE, TRUTH


def const(c):
    return lambda r: jnp.full_like(jnp.asarray(r, dtype=float), c)


@pytest.mark.parametrize("d", [1, 2, 3, 5])
@pytest.mark.parametrize("c", [0.6, 1.0, 1.7])
def test_constant_coefficient_matches_gibbs_closed_form(d, c):
    """For xi == c, eq (3.5) integrates to q(s) = a exp(-c (s - v_star)) exactly.

    So pi_xi is the Gibbs density proportional to exp(-c V), and for the OU
    potential the normalisation (3.6) pins a_xi = (c/pi)^{d/2}, Remark 3.6.
    """
    pot = OUPotential(d)
    prof = solve_profile(const(c), pot, 0.8 * c, 1.25 * c)

    a_closed = (c / math.pi) ** (d / 2)
    assert float(profile_a(prof)) == pytest.approx(a_closed, rel=1e-8)

    s = jnp.linspace(1.0, 1.0 + 8.0 / c, 200)
    q = jnp.exp(profile_log_q(prof, s))
    q_closed = a_closed * jnp.exp(-c * (s - 1.0))
    assert float(jnp.max(jnp.abs(q / q_closed - 1.0))) < 1e-8


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("which", [2, 3])
def test_normalisation_holds_for_nonconstant_xi(d, which):
    """int_{v_star}^inf q_xi(s) m_V(s) ds = 1, eq (3.6)."""
    kappa, K = ENVELOPE[which]
    pot = OUPotential(d)
    prof = solve_profile(TRUTH[which], pot, kappa, K)
    assert float(jnp.abs(profile_mass(prof) - 1.0)) < 1e-12


@pytest.mark.parametrize("which", [1, 3])
def test_pi_integrates_to_one_in_d1_by_independent_quadrature(which):
    """pi_xi integrates to 1 over R, on a grid the solver never sees."""
    kappa, K = ENVELOPE[which]
    pot = OUPotential(1)
    prof = solve_profile(TRUTH[which], pot, kappa, K)

    R = 7.5
    n = 4001
    x = jnp.linspace(-R, R, n)[:, None]
    w = simpson_weights(n, 2 * R / (n - 1))
    mass = jnp.sum(w * profile_pi(prof, pot, x))
    assert float(jnp.abs(mass - 1.0)) < 1e-9


@pytest.mark.parametrize("which", [1, 3])
def test_pi_integrates_to_one_in_d2_by_independent_quadrature(which):
    """Same in d = 2, by a tensor-product rule on a Cartesian box."""
    kappa, K = ENVELOPE[which]
    pot = OUPotential(2)
    prof = solve_profile(TRUTH[which], pot, kappa, K)

    R = 7.5
    n = 601
    g = jnp.linspace(-R, R, n)
    w = simpson_weights(n, 2 * R / (n - 1))
    xx, yy = jnp.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([xx, yy], axis=-1)
    mass = jnp.sum(w[:, None] * w[None, :] * profile_pi(prof, pot, pts))
    assert float(jnp.abs(mass - 1.0)) < 1e-8


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("which", [2, 3])
def test_gaussian_envelope_of_remark_3_6(d, which):
    """a exp(-K u) <= q_xi <= a exp(-kappa u), eq (3.7), and the bracket on a, eq (3.8).

    For the OU potential (3.8) reads (kappa/pi)^{d/2} <= a_xi <= (K/pi)^{d/2}.
    """
    kappa, K = ENVELOPE[which]
    pot = OUPotential(d)
    prof = solve_profile(TRUTH[which], pot, kappa, K)

    a = float(profile_a(prof))
    assert (kappa / math.pi) ** (d / 2) <= a <= (K / math.pi) ** (d / 2)

    s = jnp.linspace(1.0, 1.0 + 30.0, 600)
    q = jnp.exp(profile_log_q(prof, s))
    lower = a * jnp.exp(-K * (s - 1.0))
    upper = a * jnp.exp(-kappa * (s - 1.0))
    assert bool(jnp.all(q >= lower * (1.0 - 1e-12)))
    assert bool(jnp.all(q <= upper * (1.0 + 1e-12)))


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_implicit_gradient_of_the_normalisation_root_find(d):
    """d log a_xi / dc = d / (2c) when xi == c, so the root find differentiates right.

    The bisection is wrapped in stop_gradient and corrected by one Newton step,
    which is the only reason a gradient reaches a_xi at all.  For the OU
    potential a_xi = (c/pi)^{d/2}, which gives the closed form above.
    """
    pot = OUPotential(d)

    def log_a(c):
        prof = solve_profile(lambda r: jnp.full_like(jnp.asarray(r), c), pot, 0.5, 2.0)
        return profile_log_q(prof, jnp.asarray(1.0))

    for c in (0.7, 1.0, 1.6):
        g = float(jax.grad(log_a)(jnp.asarray(c)))
        assert g == pytest.approx(d / (2.0 * c), rel=1e-6)
