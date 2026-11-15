"""d=12, k=3: MCMC mean energy vs brute-force Boltzmann mean on X_k^d."""

import jax
import jax.numpy as jnp

from himag.mcmc import run_chain
from himag.sk import boltzmann_probs, hamiltonian, sample_goe
from himag.slice import enumerate_slice, random_slice_spin


def main():
    d, k, beta = 12, 3, 0.5
    key = jax.random.PRNGKey(0)
    kJ, kh, kx, kc = jax.random.split(key, 4)
    J = sample_goe(kJ, d)
    h = jax.random.normal(kh, (d,)) * 0.25
    states = enumerate_slice(d, k)
    pi = boltzmann_probs(states, J, h, beta)
    E = hamiltonian(states, J, h)
    exact = float(jnp.sum(pi * E))
    x0 = random_slice_spin(kx, d, k)
    samples = run_chain(kc, x0, J, h, beta, n_steps=8000)
    emp = float(jnp.mean(hamiltonian(samples[1000:], J, h)))
    print(f"d={d} k={k} beta={beta} n_slice={int(states.shape[0])}")
    print(f"exact Boltzmann mean energy: {exact:.6f}")
    print(f"MCMC mean energy (after burn-in): {emp:.6f}")
    print(f"abs gap: {abs(emp - exact):.6f}")


if __name__ == "__main__":
    main()
