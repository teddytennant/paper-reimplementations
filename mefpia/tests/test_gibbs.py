"""Gibbs map (eq 30) is a valid density and unique free-energy minimizer (Thm 2)."""

import jax
import jax.numpy as jnp

from mefpia.algo import gibbs_state
from mefpia.ops import is_density, random_density, von_neumann_entropy

from helpers import random_hermitian


def _free_energy(h, sigma, temperature):
    return jnp.trace(h @ sigma).real - temperature * von_neumann_entropy(sigma)


def test_gibbs_is_density_and_unique_minimizer():
    key = jax.random.PRNGKey(2)
    key_h, key_s = jax.random.split(key)
    h = random_hermitian(key_h, 2)
    temperature = 0.7
    rho = gibbs_state(h, temperature)
    assert bool(is_density(rho, atol=1e-6))
    assert rho.dtype == h.dtype
    f_star = _free_energy(h, rho, temperature)
    keys = jax.random.split(key_s, 8)
    for k in keys:
        sigma = random_density(k, 2)
        f_sigma = _free_energy(h, sigma, temperature)
        assert f_star <= f_sigma + 1e-6
