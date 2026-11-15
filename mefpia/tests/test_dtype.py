"""Shape and dtype stay consistent for complex64 and complex128."""

import jax
import jax.numpy as jnp
import pytest

from mefpia.algo import gibbs_state, mefpia
from mefpia.ops import random_density
from mefpia.payoff import as_tensor, effective_hamiltonian, payoff
from mefpia.pd import prisoners_dilemma

from helpers import random_hermitian


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_complex_dtype_consistent(dtype):
    dims = (2, 2)
    key = jax.random.PRNGKey(5)
    key_r, key_a, key_b = jax.random.split(key, 3)
    r_mat = random_hermitian(key_r, 4, dtype=dtype).astype(dtype)
    r_ten = as_tensor(r_mat, dims).astype(dtype)
    rhos = [
        random_density(key_a, 2, dtype=dtype).astype(dtype),
        random_density(key_b, 2, dtype=dtype).astype(dtype),
    ]
    r = payoff(r_ten, rhos)
    assert r.dtype == jnp.finfo(dtype).dtype
    h = effective_hamiltonian(r_ten, rhos, 0)
    assert h.dtype == dtype
    assert h.shape == (2, 2)
    rho_g = gibbs_state(h, 1.0)
    assert rho_g.dtype == dtype
    assert rho_g.shape == (2, 2)
    out, info = mefpia( [r_ten, r_ten], rhos, t0=1.0, alpha=0.9, max_steps=3)
    for rho in out:
        assert rho.dtype == dtype
        assert rho.shape == (2, 2)
    assert info["payoffs"].dtype == jnp.finfo(dtype).dtype


def test_pd_reward_shapes():
    game = prisoners_dilemma(dtype=jnp.complex64)
    r1, r2 = game["rewards"]
    assert r1.shape == (2, 2, 2, 2)
    assert r2.shape == (2, 2, 2, 2)
    assert r1.dtype == jnp.complex64
    assert game["r1_matrix"].shape == (4, 4)
