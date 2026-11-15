"""Unitarity and density-matrix invariants for the variational blocks."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_vqt.circuit import (
    apply_gate,
    ket0_dm,
    kron_dm,
    partial_trace,
    rot,
    rx,
    ry,
    rz,
    sel_ranges,
    strongly_entangling_layers,
)


def _is_unitary(u, atol=1e-10):
    n = u.shape[0]
    return jnp.allclose(u.conj().T @ u, jnp.eye(n), atol=atol)


def _is_dm(rho, atol=1e-8):
    herm = jnp.allclose(rho, rho.conj().T, atol=atol)
    tr = jnp.allclose(jnp.trace(rho), 1.0, atol=atol)
    evals = jnp.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    psd = jnp.all(evals >= -atol)
    return herm and tr and psd


def test_single_qubit_gates_unitary():
    rng = np.random.default_rng(0)
    for theta in rng.uniform(0, 2 * np.pi, size=5):
        assert _is_unitary(rx(theta))
        assert _is_unitary(ry(theta))
        assert _is_unitary(rz(theta))
    for _ in range(5):
        phi, th, om = rng.uniform(0, 2 * np.pi, size=3)
        assert _is_unitary(rot(phi, th, om))
        # PennyLane Rot = RZ(φ) RY(θ) RZ(ω)
        assert jnp.allclose(rot(phi, th, om), rz(phi) @ ry(th) @ rz(om), atol=1e-12)


def test_cnot_on_computational_basis():
    cnot = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=jnp.complex128,
    )
    assert _is_unitary(cnot)
    # |10⟩ (qubit 0 = 1) flips target qubit 1 → |11⟩
    rho = jnp.zeros((4, 4), dtype=jnp.complex128).at[2, 2].set(1.0)
    out = apply_gate(rho, cnot, (0, 1), 2)
    assert jnp.allclose(jnp.diag(out).real, jnp.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)


def test_sel_ranges_match_pennylane():
    # r_l = (l mod (n-1)) + 1
    assert sel_ranges(6, 1) == (1,)
    assert sel_ranges(6, 5) == (1, 2, 3, 4, 5)
    assert sel_ranges(4, 5) == (1, 2, 3, 1, 2)
    assert sel_ranges(2, 4) == (1, 1, 1, 1)


def test_strongly_entangling_layers_preserve_purity():
    key = jax.random.PRNGKey(1)
    n, layers = 4, 2
    weights = jax.random.uniform(key, (layers, n, 3), minval=0.0, maxval=2 * jnp.pi)
    rho0 = ket0_dm(n)
    rho = strongly_entangling_layers(rho0, weights, n)
    assert _is_dm(rho)
    purity = jnp.real(jnp.trace(rho @ rho))
    assert jnp.allclose(purity, 1.0, atol=1e-8)


def test_partial_trace_of_product_state():
    a = ket0_dm(1)
    # RX(π)|0⟩ = −i|1⟩, so qubit-1 reduced state is |1⟩⟨1|
    x = rx(jnp.pi)
    ket1 = x @ jnp.array([1, 0], dtype=jnp.complex128)
    b = jnp.outer(ket1, ket1.conj())
    rho = kron_dm(a, b)
    keep0 = partial_trace(rho, 2, keep=(0,))
    keep1 = partial_trace(rho, 2, keep=(1,))
    assert jnp.allclose(keep0, a, atol=1e-10)
    assert jnp.allclose(keep1, b, atol=1e-10)


def test_reduced_state_after_sel_is_valid_dm():
    key = jax.random.PRNGKey(2)
    n = 6
    weights = jax.random.uniform(key, (1, n, 3), minval=0.0, maxval=2 * jnp.pi)
    rho = strongly_entangling_layers(ket0_dm(n), weights, n)
    red = partial_trace(rho, n, keep=(0, 1))
    assert red.shape == (4, 4)
    assert _is_dm(red)
    purity = jnp.real(jnp.trace(red @ red))
    assert purity <= 1.0 + 1e-8
