"""MEFPIA and MMWU keep valid density matrices (Algorithm 1)."""

import jax
import jax.numpy as jnp

from mefpia.algo import mefpia, mmwu_step
from mefpia.ops import is_density
from mefpia.payoff import as_tensor

from helpers import random_hermitian, random_product_state


def test_mefpia_and_mmwu_preserve_density():
    dims = (2, 2)
    key = jax.random.PRNGKey(3)
    key_r1, key_r2, key_s = jax.random.split(key, 3)
    rewards = (
        as_tensor(random_hermitian(key_r1, 4), dims),
        as_tensor(random_hermitian(key_r2, 4), dims),
    )
    rhos0 = random_product_state(key_s, dims)
    rhos, info = mefpia(
        rewards, rhos0, t0=1.0, alpha=0.98, max_steps=12, tol=0.0
    )
    assert info["steps"] == 12
    assert info["payoffs"].shape == (13, 2)
    for rho in rhos:
        assert bool(is_density(rho, atol=1e-5))
        assert rho.shape == (2, 2)
        assert rho.dtype == rhos0[0].dtype

    mmwu = list(rhos0)
    for _ in range(5):
        mmwu = mmwu_step(rewards, mmwu, gamma=0.2)
    for rho in mmwu:
        assert bool(is_density(rho, atol=1e-5))
