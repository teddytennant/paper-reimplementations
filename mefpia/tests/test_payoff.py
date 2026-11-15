"""Tensor-contraction payoff (eq 23) vs an explicit joint trace."""

import jax
import jax.numpy as jnp
import pytest

from mefpia.payoff import as_matrix, as_tensor, payoff

from helpers import kron_all, random_hermitian, random_product_state


@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)])
def test_tensor_payoff_matches_joint_trace(dims):
    n = len(dims)
    joint_dim = 1
    for d in dims:
        joint_dim *= d
    key = jax.random.PRNGKey(0)
    key_r, key_s = jax.random.split(key)
    r_mat = random_hermitian(key_r, joint_dim)
    r_ten = as_tensor(r_mat, dims)
    rhos = random_product_state(key_s, dims)
    contracted = payoff(r_ten, rhos)
    joint = jnp.trace(r_mat @ kron_all(rhos))
    assert contracted.dtype == jnp.float64
    assert contracted.shape == ()
    assert jnp.allclose(contracted, joint.real, atol=1e-5)
    assert jnp.allclose(as_matrix(r_ten), r_mat, atol=1e-12)
