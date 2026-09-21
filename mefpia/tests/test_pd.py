"""Prisoner's Dilemma mini-experiment 1 (Section VII, eqs 33-36, 28)."""

import jax
import jax.numpy as jnp

from mefpia.algo import mefpia
from mefpia.ops import is_density, random_density
from mefpia.payoff import effective_hamiltonian
from mefpia.pd import prisoners_dilemma, relative_payoff_error


def test_pd_mefpia_reduces_relative_error_and_nash_gap():
    game = prisoners_dilemma(dtype=jnp.complex128)
    rewards = game["rewards"]
    r_star = game["r_star"]
    # Start near cooperate so mutual defection is a clear move of the payoff.
    rho0 = jnp.asarray([[0.85, 0.0], [0.0, 0.15]], dtype=jnp.complex128)
    rhos, info = mefpia(
        rewards, [rho0, rho0], t0=1.0, alpha=0.98, max_steps=100, tol=1e-10
    )
    for rho in rhos:
        assert bool(is_density(rho, atol=1e-5))

    r0 = info["payoffs"][0]
    r_end = info["payoffs"][-1]
    err0 = relative_payoff_error(r0, r_star)
    err_end = relative_payoff_error(r_end, r_star)
    assert jnp.all(err_end < err0)
    assert jnp.all(err_end < 0.2)

    keys = jax.random.split(jax.random.PRNGKey(4), 4)
    for i, rho in enumerate(rhos):
        h = effective_hamiltonian(rewards[i], rhos, i)
        gap = jnp.trace(h @ rho).real - jnp.linalg.eigvalsh(h).real[0]
        assert gap <= 5e-2
        sigma = random_density(keys[i], 2)
        nash = jnp.trace(h @ (rho - sigma)).real
        assert nash <= 5e-2

    # Mutual defection |1><1| is the classical Nash; mass on |1> should rise.
    assert rhos[0][1, 1].real > rho0[1, 1].real
    assert rhos[1][1, 1].real > rho0[1, 1].real
