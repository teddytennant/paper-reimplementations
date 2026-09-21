"""Gibbs map, MEFPIA (Algorithm 1), and a Kale-style MMWU step."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from mefpia.ops import hermitize
from mefpia.payoff import effective_hamiltonian, payoffs

Array = jnp.ndarray


def gibbs_state(hamiltonian: Array, temperature: float | Array) -> Array:
    """Unique minimizer of Tr(H sigma) - T S(sigma) (Thm 2, eq 30).

    M = exp(-H / T), rho = M / Tr(M). Uses a Hermitian eigendecomposition
    with a max-eigenvalue shift so the exponential does not overflow at
    small T (paper eqs 39-40).
    """
    h = hermitize(hamiltonian)
    evals, evecs = jnp.linalg.eigh(h)
    scaled = -evals.real / temperature
    weights = jnp.exp(scaled - jnp.max(scaled))
    weights = weights / jnp.sum(weights)
    rho = (evecs * weights.astype(evecs.dtype)) @ jnp.conjugate(evecs.T)
    return hermitize(rho)


def _frobenius(a: Array, b: Array) -> Array:
    d = a - b
    return jnp.sqrt(jnp.real(jnp.vdot(d, d)))


def mefpia(
    reward_tensors: tuple[Array, ...] | list[Array],
    rhos: tuple[Array, ...] | list[Array],
    *,
    t0: float = 1.0,
    alpha: float = 0.98,
    max_steps: int = 100,
    tol: float = 1e-8,
) -> tuple[list[Array], dict[str, Any]]:
    """Simultaneous MEFPIA with geometric cooling T_k = T0 * alpha^k.

    Algorithm 1 and eqs 30-31. All players update from the same product
    state. Stops when ||rho^{(i)[k+1]} - rho^{(i)[k]}||_F <= tol for every i.
    """
    current = [hermitize(rho) for rho in rhos]
    history_payoffs = [payoffs(reward_tensors, current)]
    history_t = []
    converged = False
    steps = 0
    for k in range(max_steps):
        temp = t0 * (alpha ** k)
        history_t.append(temp)
        hamils = [
            effective_hamiltonian(reward_tensors[i], current, i)
            for i in range(len(current))
        ]
        nxt = [gibbs_state(h, temp) for h in hamils]
        diffs = [_frobenius(n, o) for n, o in zip(nxt, current)]
        current = nxt
        steps = k + 1
        history_payoffs.append(payoffs(reward_tensors, current))
        if all(float(d) <= tol for d in diffs):
            converged = True
            break
    info = {
        "steps": steps,
        "converged": converged,
        "temperatures": jnp.asarray(history_t),
        "payoffs": jnp.stack(history_payoffs),
        "rhos": current,
    }
    return current, info


def mmwu_step(
    reward_tensors: tuple[Array, ...] | list[Array],
    rhos: tuple[Array, ...] | list[Array],
    gamma: float,
) -> list[Array]:
    """One Kale-style matrix MWU step: rho <- normalize(exp(log rho - gamma H)).

    H is the effective Hamiltonian of eq 26, so the update descends the
    energy Tr(H rho) and is comparable to a finite-temperature MEFPIA step.
    """
    current = [hermitize(rho) for rho in rhos]
    updated = []
    for i, rho in enumerate(current):
        h = effective_hamiltonian(reward_tensors[i], current, i)
        evals, evecs = jnp.linalg.eigh(rho)
        log_evals = jnp.log(jnp.clip(evals.real, 1e-12, None))
        log_rho = (evecs * log_evals.astype(evecs.dtype)) @ jnp.conjugate(evecs.T)
        generator = hermitize(log_rho - gamma * h)
        ge, gv = jnp.linalg.eigh(generator)
        weights = jnp.exp(ge.real - jnp.max(ge.real))
        weights = weights / jnp.sum(weights)
        nxt = (gv * weights.astype(gv.dtype)) @ jnp.conjugate(gv.T)
        updated.append(hermitize(nxt))
    return updated
