"""JAX density-matrix simulator and PennyLane-style strongly entangling layers.

Used to realize the variational blocks of arXiv:2609.18565. Qubit 0 is the
leftmost factor in the tensor product (most significant bit of the basis index).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


def cdtype():
    return jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64


def rx(theta):
    """RX(θ) = cos(θ/2) I − i sin(θ/2) X."""
    theta = jnp.asarray(theta)
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.stack(
        [jnp.stack([c, -1j * s]), jnp.stack([-1j * s, c])]
    ).astype(cdtype())


def ry(theta):
    """RY(θ) = cos(θ/2) I − i sin(θ/2) Y."""
    theta = jnp.asarray(theta)
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.stack(
        [jnp.stack([c, -s]), jnp.stack([s, c])]
    ).astype(cdtype())


def rz(theta):
    """RZ(θ) = diag(e^{−iθ/2}, e^{iθ/2})."""
    theta = jnp.asarray(theta)
    phase = 1j * theta / 2.0
    z = jnp.zeros((), dtype=cdtype())
    return jnp.stack(
        [jnp.stack([jnp.exp(-phase), z]), jnp.stack([z, jnp.exp(phase)])]
    ).astype(cdtype())


def rot(phi, theta, omega):
    """PennyLane Rot(φ, θ, ω) = RZ(φ) RY(θ) RZ(ω)."""
    return rz(phi) @ ry(theta) @ rz(omega)


def ket0_dm(n: int):
    dim = 1 << n
    rho = jnp.zeros((dim, dim), dtype=cdtype())
    return rho.at[0, 0].set(jnp.asarray(1.0, dtype=cdtype()))


def kron_dm(rho_a, rho_b):
    """Tensor product of density matrices, batched over leading axes."""
    da = rho_a.shape[-1]
    db = rho_b.shape[-1]
    out = jnp.einsum("...ij,...kl->...ikjl", rho_a, rho_b)
    return out.reshape(rho_a.shape[:-2] + (da * db, da * db))


def hermitize(rho, eps: float = 1e-12):
    """Project onto Hermitian trace-1 matrices (numerical cleanup)."""
    rho = 0.5 * (rho + jnp.swapaxes(jnp.conjugate(rho), -1, -2))
    tr = jnp.trace(rho, axis1=-2, axis2=-1).real
    tr = jnp.maximum(tr, eps)
    return rho / tr[..., None, None]


def apply_gate(rho, gate, qubits: Sequence[int], n: int):
    """Left-right apply `gate` on `qubits` of an n-qubit density matrix.

    `rho` has shape (..., 2^n, 2^n). `gate` is (2^k, 2^k) for k = len(qubits).
    """
    qubits = tuple(int(q) for q in qubits)
    k = len(qubits)
    rest = tuple(i for i in range(n) if i not in qubits)
    ket_order = qubits + rest
    n_batch = rho.ndim - 2
    perm = (
        tuple(range(n_batch))
        + tuple(n_batch + q for q in ket_order)
        + tuple(n_batch + n + q for q in ket_order)
    )
    rho_t = rho.reshape(rho.shape[:-2] + (2,) * (2 * n))
    rho_t = jnp.transpose(rho_t, perm)
    g = 1 << k
    r = 1 << (n - k)
    rho_t = rho_t.reshape(rho.shape[:-2] + (g, r, g, r))
    gate = jnp.asarray(gate, dtype=rho.dtype)
    rho_t = jnp.einsum("ai,...ibjd,cj->...abcd", gate, rho_t, jnp.conjugate(gate))
    rho_t = rho_t.reshape(rho.shape[:-2] + (2,) * (2 * n))
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    rho_t = jnp.transpose(rho_t, tuple(inv))
    return rho_t.reshape(rho.shape)


def partial_trace(rho, n: int, keep: Sequence[int]):
    """Trace out all qubits except `keep` (order of `keep` is the output order)."""
    keep = tuple(int(q) for q in keep)
    traced = tuple(i for i in range(n) if i not in keep)
    k = len(keep)
    t = n - k
    n_batch = rho.ndim - 2
    perm = (
        tuple(range(n_batch))
        + tuple(n_batch + i for i in keep)
        + tuple(n_batch + i for i in traced)
        + tuple(n_batch + n + i for i in keep)
        + tuple(n_batch + n + i for i in traced)
    )
    rho_t = rho.reshape(rho.shape[:-2] + (2,) * (2 * n))
    rho_t = jnp.transpose(rho_t, perm)
    dim_k = 1 << k
    dim_t = 1 << t
    rho_t = rho_t.reshape(rho.shape[:-2] + (dim_k, dim_t, dim_k, dim_t))
    out = jnp.trace(rho_t, axis1=-3, axis2=-1)
    return hermitize(out)


def sel_ranges(n_wires: int, n_layers: int) -> tuple[int, ...]:
    """PennyLane StronglyEntanglingLayers default ranges.

    r_l = (l mod (n−1)) + 1 for n > 1, else 0.
    """
    if n_wires <= 1:
        return tuple(0 for _ in range(n_layers))
    return tuple((l % (n_wires - 1)) + 1 for l in range(n_layers))


def strongly_entangling_layers(rho, weights, n: int):
    """Apply PennyLane StronglyEntanglingLayers to a density matrix.

    `weights` has shape (L, n, 3): Rot(φ, θ, ω) on each qubit, then circular
    CNOTs with range r_l.
    """
    n_layers = int(weights.shape[0])
    ranges = sel_ranges(n, n_layers)
    cnot = jnp.asarray(CNOT, dtype=cdtype())
    for layer in range(n_layers):
        for q in range(n):
            phi, theta, omega = weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2]
            rho = apply_gate(rho, rot(phi, theta, omega), (q,), n)
        r = ranges[layer]
        if n > 1 and r:
            for q in range(n):
                rho = apply_gate(rho, cnot, (q, (q + r) % n), n)
    return rho
