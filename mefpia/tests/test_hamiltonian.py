"""Effective Hamiltonian identities (Def 1, eqs 26-27)."""

import jax
import jax.numpy as jnp
import pytest

from mefpia.ops import is_hermitian
from mefpia.payoff import as_tensor, effective_hamiltonian, payoff

from helpers import random_hermitian, random_product_state


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)])
def test_effective_hamiltonian_eq27(dims):
    n = len(dims)
    joint_dim = 1
    for d in dims:
        joint_dim *= d
    key = jax.random.PRNGKey(1)
    key_r, key_s = jax.random.split(key)
    r_mat = random_hermitian(key_r, joint_dim)
    r_ten = as_tensor(r_mat, dims)
    rhos = random_product_state(key_s, dims)
    r_i = payoff(r_ten, rhos)
    for player in range(n):
        h = effective_hamiltonian(r_ten, rhos, player)
        assert h.shape == (dims[player], dims[player])
        assert bool(is_hermitian(h, atol=1e-8))
        energy = jnp.trace(rhos[player] @ h)
        assert jnp.allclose(r_i, -energy.real, atol=1e-5)
        assert jnp.abs(energy.imag) < 1e-8
