"""Tensor-contraction payoffs and effective Hamiltonians (eqs 23, 25-27)."""

from __future__ import annotations

import string

import jax.numpy as jnp

from mefpia.ops import hermitize

Array = jnp.ndarray
_LETTERS = string.ascii_lowercase


def as_tensor(reward_matrix: Array, dims: tuple[int, ...]) -> Array:
    """Reshape a joint operator on tensor(C^{d_i}) into axes (d1..dN, d1..dN)."""
    return jnp.reshape(reward_matrix, dims + dims)


def as_matrix(reward_tensor: Array) -> Array:
    """Flatten a reward tensor back to a joint matrix in Kronecker order."""
    n = reward_tensor.ndim // 2
    dim = 1
    for d in reward_tensor.shape[:n]:
        dim *= int(d)
    return jnp.reshape(reward_tensor, (dim, dim))


def _n_players(reward_tensor: Array) -> int:
    ndim = reward_tensor.ndim
    if ndim % 2 != 0:
        raise ValueError("reward tensor must have an even number of axes")
    n = ndim // 2
    if 2 * n > len(_LETTERS):
        raise ValueError("too many players for einsum letter budget")
    return n


def payoff(reward_tensor: Array, rhos: tuple[Array, ...] | list[Array]) -> Array:
    """Scalar payoff r_i = sum_{j,j'} R_{j,j'} prod_k rho^{(k)}_{j'_k, j_k} (eq 23).

    Complexity is one einsum over the 2N reward axes and N density matrices,
    O((N+1) prod_i d_i^2), without forming the joint density matrix.
    """
    n = _n_players(reward_tensor)
    if len(rhos) != n:
        raise ValueError(f"expected {n} density matrices, got {len(rhos)}")
    row = _LETTERS[:n]
    col = _LETTERS[n : 2 * n]
    rho_subs = [col[i] + row[i] for i in range(n)]
    spec = f"{row}{col},{','.join(rho_subs)}->"
    value = jnp.einsum(spec, reward_tensor, *rhos, optimize=True)
    return jnp.real(value)


def payoffs(
    reward_tensors: tuple[Array, ...] | list[Array],
    rhos: tuple[Array, ...] | list[Array],
) -> Array:
    """Stack of per-player payoffs (eq 17 via eq 23)."""
    return jnp.stack([payoff(r, rhos) for r in reward_tensors])


def effective_hamiltonian(
    reward_tensor: Array,
    rhos: tuple[Array, ...] | list[Array],
    player: int,
) -> Array:
    """H^{(i)} = -Tr_{-i}(R_i rho_{-i}^{(s)}) (Def 1, eqs 25-26).

    Leaves player i's two axes uncontracted. Then r_i = -Tr(rho^{(i)} H^{(i)})
    (eq 27). The result is hermitized so later matrix exponentials stay on
    Hermitian generators.
    """
    n = _n_players(reward_tensor)
    if len(rhos) != n:
        raise ValueError(f"expected {n} density matrices, got {len(rhos)}")
    if not 0 <= player < n:
        raise ValueError(f"player {player} out of range for N={n}")
    row = _LETTERS[:n]
    col = _LETTERS[n : 2 * n]
    others = []
    other_rhos = []
    for i in range(n):
        if i == player:
            continue
        others.append(col[i] + row[i])
        other_rhos.append(rhos[i])
    out = row[player] + col[player]
    if others:
        spec = f"{row}{col},{','.join(others)}->{out}"
        contracted = jnp.einsum(spec, reward_tensor, *other_rhos, optimize=True)
    else:
        contracted = reward_tensor
    return hermitize(-contracted)
