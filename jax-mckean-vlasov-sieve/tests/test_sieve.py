"""The ReQU sieve of eq (4.1): the structural bounds, and what ReQU buys."""

import jax
import jax.numpy as jnp
import pytest

from mkv_sieve import clamp_tau, init_requ_params, make_xi, requ, requ_net
from mkv_sieve.potential import OUPotential
from mkv_sieve.sieve import domain_bound

KAPPA, K, TAU, U = 0.8, 1.2, 0.05, 1.0


def scaled_params(key, scale, width=5):
    ks = jax.random.split(key, 4)
    return [
        (scale * jax.random.normal(ks[0], (width, 1)), scale * jax.random.normal(ks[1], (width,))),
        (scale * jax.random.normal(ks[2], (1, width)), scale * jax.random.normal(ks[3], (1,))),
    ]


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e3, 1e6, 1e12])
def test_structural_bounds_hold_for_any_parameters(scale):
    """kappa <= xi_theta <= K everywhere, Assumption 3.3, whatever theta is.

    The bounds come from the clamp S_tau in eq (4.1), not from the optimiser
    staying somewhere sensible, so they have to survive parameters no optimiser
    would produce.
    """
    r = jnp.concatenate([jnp.zeros(1), jnp.logspace(-8, 2, 200)])
    for seed in range(6):
        params = scaled_params(jax.random.PRNGKey(seed), scale)
        v = make_xi(params, KAPPA, K, TAU, U)(r)
        assert bool(jnp.all(jnp.isfinite(v)))
        assert float(jnp.min(v)) >= KAPPA
        assert float(jnp.max(v)) <= K


def test_bounds_hold_for_degenerate_parameters():
    zeros = [(jnp.zeros((4, 1)), jnp.zeros(4)), (jnp.zeros((1, 4)), jnp.zeros(1))]
    r = jnp.linspace(0.0, 5.0, 50)
    v = make_xi(zeros, KAPPA, K, TAU, U)(r)
    assert bool(jnp.all((v >= KAPPA) & (v <= K)))


def test_gradients_are_finite_in_the_saturated_region():
    """S_tau never has an exactly flat branch, so a saturated unit still gets a signal."""
    for z in (-1e6, -50.0, 0.0, 0.5, 1.0, 50.0, 1e6):
        g = float(jax.grad(clamp_tau)(jnp.asarray(z), TAU))
        assert jnp.isfinite(g)
    assert float(jax.grad(clamp_tau)(jnp.asarray(-50.0), TAU)) >= 0.0


def test_clamp_is_identity_on_its_window_and_1_lipschitz():
    z = jnp.linspace(TAU, 1.0 - TAU, 101)
    assert float(jnp.max(jnp.abs(clamp_tau(z, TAU) - z))) == 0.0
    zz = jnp.linspace(-30.0, 30.0, 200001)
    s = clamp_tau(zz, TAU)
    slope = jnp.diff(s) / jnp.diff(zz)
    assert float(jnp.max(slope)) <= 1.0 + 1e-12
    assert float(jnp.min(slope)) >= -1e-15
    assert float(jnp.min(s)) >= 0.0 and float(jnp.max(s)) <= 1.0
    # strictly interior wherever the exponential tails have not underflowed,
    # which at tau = 0.05 means roughly 1.6 outside the window
    mid = clamp_tau(jnp.linspace(-1.4, 2.3, 2001), TAU)
    assert float(jnp.min(mid)) > 0.0 and float(jnp.max(mid)) < 1.0


def test_requ_represents_a_quadratic_exactly():
    """z^2 = sigma(z) + sigma(-z) with sigma(z) = (z_+)^2.

    This is why Section 4.1 picks ReQU: the sieve elements have to be C^1 for
    the linearised forward map of Section 4.2, and a quadratic costs two units.
    """
    params = [
        (jnp.array([[1.0], [-1.0]]), jnp.zeros(2)),
        (jnp.array([[1.0, 1.0]]), jnp.zeros(1)),
    ]
    z = jnp.linspace(-4.0, 4.0, 1001)
    assert float(jnp.max(jnp.abs(requ_net(params, z) - z * z))) == 0.0


def test_requ_represents_an_affine_function_exactly():
    """z = (sigma(z + c) - sigma(c - z)) / (4c) on |z| < c, the paper's two units."""
    c = 8.0
    params = [
        (jnp.array([[1.0], [-1.0]]), jnp.array([c, c])),
        (jnp.array([[1.0 / (4 * c), -1.0 / (4 * c)]]), jnp.zeros(1)),
    ]
    z = jnp.linspace(-c + 1e-6, c - 1e-6, 1001)
    assert float(jnp.max(jnp.abs(requ_net(params, z) - z))) < 1e-14


def test_requ_activation_and_its_derivative():
    z = jnp.array([-2.0, -1e-9, 0.0, 1e-9, 3.0])
    assert jnp.allclose(requ(z), jnp.where(z > 0, z * z, 0.0))
    g = jax.vmap(jax.grad(lambda t: requ(t)))(z)
    assert jnp.allclose(g, jnp.where(z > 0, 2 * z, 0.0))


def test_xi_is_constant_above_the_domain_bound():
    """For r > U the network is extended by xi_theta(U), Section 4.1."""
    params = init_requ_params(jax.random.PRNGKey(0), width=4, U=U)
    xi = make_xi(params, KAPPA, K, TAU, U)
    at_u = float(xi(jnp.asarray(U)))
    beyond = xi(jnp.array([U + 1e-6, 2.0, 10.0, 1e6]))
    assert float(jnp.max(jnp.abs(beyond - at_u))) == 0.0


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_domain_bound_is_max_one_and_a_plus(d):
    """U >= max(1, a_+) with a_+ = Z_K^{-1} = (K/pi)^{d/2}, Section 4.1."""
    pot = OUPotential(d)
    assert domain_bound(pot, K) >= max(1.0, (K / jnp.pi) ** (d / 2))
