"""Proposition 3.4 itself: pi_xi solves the stationary nonlinear Fokker-Planck equation.

The stationary form of the equation in Section 1 is

    div( grad pi + grad V(x) xi(pi(x)) pi(x) ) = 0,

and Proposition 3.4 gets it from the stronger zero-flux identity
grad pi = -pi xi(pi) grad V.  Both are checked here against the density the
solver actually produces, differentiated with jax.grad, so what is being tested
is the ODE plus the root find plus the interpolant, not the algebra.

The tolerances are the solver's own discretisation error.  Each check is run at
two grid resolutions and the residual has to shrink at the interpolant's order,
which is what says it is discretisation and not a mistake.
"""

import jax
import jax.numpy as jnp
import pytest

from mkv_sieve import OUPotential, SolverConfig, profile_log_pi, solve_profile
from mkv_sieve.truth import ENVELOPE, TRUTH


def ray(d, n=40, lo=0.05, hi=2.5):
    r = jnp.linspace(lo, hi, n)
    e1 = jnp.zeros((n, d)).at[:, 0].set(1.0)
    return r[:, None] * e1


def flux_residual(which, d, n_y):
    kappa, K = ENVELOPE[which]
    xi = TRUTH[which]
    pot = OUPotential(d)
    prof = solve_profile(xi, pot, kappa, K, SolverConfig(n_y=n_y))

    def pi(x):
        return jnp.exp(profile_log_pi(prof, pot, x))

    x = ray(d)
    p = pi(x)
    grad_pi = jax.vmap(jax.grad(pi))(x)
    flux = grad_pi + pot.grad_V(x) * (xi(p) * p)[:, None]
    scale = jnp.linalg.norm(grad_pi, axis=-1)
    return float(jnp.max(jnp.linalg.norm(flux, axis=-1) / scale))


def divergence_residual(which, d, n_y):
    kappa, K = ENVELOPE[which]
    xi = TRUTH[which]
    pot = OUPotential(d)
    prof = solve_profile(xi, pot, kappa, K, SolverConfig(n_y=n_y))

    def pi(x):
        return jnp.exp(profile_log_pi(prof, pot, x))

    def flux(x):
        p = pi(x)
        return jax.grad(pi)(x) + pot.grad_V(x) * xi(p) * p

    x = ray(d)
    div = jax.vmap(lambda z: jnp.trace(jax.jacfwd(flux)(z)))(x)
    lap = jax.vmap(lambda z: jnp.trace(jax.hessian(pi)(z)))(x)
    return float(jnp.max(jnp.abs(div))), float(jnp.max(jnp.abs(lap)))


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("which", [2, 3])
def test_zero_flux_identity(d, which):
    """grad pi + grad V xi(pi) pi = 0 pointwise, the identity Proposition 3.4 proves."""
    assert flux_residual(which, d, 6001) < 5e-9


@pytest.mark.parametrize("which", [2, 3])
def test_zero_flux_residual_is_discretisation_error(which):
    """Halving the grid step cuts the residual like h^3, the Hermite slope order.

    A residual that sat still under refinement would be a modelling error, not a
    discretisation one.
    """
    coarse = flux_residual(which, 2, 1501)
    fine = flux_residual(which, 2, 3001)
    finer = flux_residual(which, 2, 6001)
    assert coarse / fine > 4.0
    assert fine / finer > 4.0


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("which", [2, 3])
def test_stationary_fokker_planck_residual(d, which):
    """div(grad pi + grad V xi(pi) pi) = 0, against the size of Laplacian pi."""
    div, lap = divergence_residual(which, d, 6001)
    assert lap > 0.1  # the terms it has to cancel are O(1), so zero is not trivial
    assert div / lap < 1e-6


@pytest.mark.parametrize("which", [2, 3])
def test_fokker_planck_residual_is_discretisation_error(which):
    """The divergence goes through a second derivative of the interpolant, so h^2."""
    coarse, _ = divergence_residual(which, 2, 1501)
    fine, _ = divergence_residual(which, 2, 3001)
    finer, _ = divergence_residual(which, 2, 6001)
    assert coarse / fine > 2.5
    assert fine / finer > 2.5
