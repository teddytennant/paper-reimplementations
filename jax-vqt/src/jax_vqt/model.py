"""Variational quantum transformer (arXiv:2609.18565).

Encoder heads, integration circuits, connector + decoder blocks, and a
two-qubit computational-basis readout. At most six qubits are active.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from jax_vqt.circuit import (
    apply_gate,
    cdtype,
    kron_dm,
    ket0_dm,
    partial_trace,
    rx,
    ry,
    strongly_entangling_layers,
)


@dataclass(frozen=True)
class VQTConfig:
    """Paper defaults: context c=4, |V|=4, two-qubit readout."""

    n_encoder: int = 2
    n_decoder: int = 2
    n_var_layers: int = 1
    context: int = 4
    vocab_size: int = 4

    @property
    def name(self) -> str:
        return f"q_e{self.n_encoder}_d{self.n_decoder}_v{self.n_var_layers}"


def parameter_count(n_encoder: int, n_decoder: int, n_var_layers: int) -> int:
    """Paper formula: P(E, D, V) = (48 + 36(E−1))V + D(18V + 2)."""
    e, d, v = int(n_encoder), int(n_decoder), int(n_var_layers)
    return (48 + 36 * (e - 1)) * v + d * (18 * v + 2)


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def init_params(key, config: VQTConfig):
    """Uniform-[0, 2π] angles for every Rot/RY (initialization unspecified)."""
    e, d, v = config.n_encoder, config.n_decoder, config.n_var_layers
    keys = jax.random.split(key, 6)
    two_pi = 2.0 * jnp.pi
    return {
        "enc0_heads": jax.random.uniform(keys[0], (2, v, 6, 3), minval=0.0, maxval=two_pi),
        "enc0_int": jax.random.uniform(keys[1], (v, 4, 3), minval=0.0, maxval=two_pi),
        "enc_heads": jax.random.uniform(
            keys[2], (max(e - 1, 0), 2, v, 4, 3), minval=0.0, maxval=two_pi
        ),
        "enc_int": jax.random.uniform(
            keys[3], (max(e - 1, 0), v, 4, 3), minval=0.0, maxval=two_pi
        ),
        "dec_ry": jax.random.uniform(keys[4], (d, 2), minval=0.0, maxval=two_pi),
        "dec_var": jax.random.uniform(keys[5], (d, v, 6, 3), minval=0.0, maxval=two_pi),
    }


def encode_context(tokens):
    """Angle encoding (paper eq. 2): |ψ_x⟩ = ⊗_j RX(π/2 x_j)|0⟩, x_j ∈ {0,1,2,3}."""
    tokens = jnp.asarray(tokens)
    thetas = tokens.astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32) * (
        jnp.pi / 2.0
    )
    r0 = _rx_on_zero(thetas[0])
    r1 = _rx_on_zero(thetas[1])
    r2 = _rx_on_zero(thetas[2])
    r3 = _rx_on_zero(thetas[3])
    rho = kron_dm(kron_dm(r0, r1), kron_dm(r2, r3))
    return rho


def encode_statevector(tokens):
    """Statevector form of eq. 2, for tests."""
    tokens = jnp.asarray(tokens)
    thetas = tokens.astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32) * (
        jnp.pi / 2.0
    )
    psi = jnp.ones((1,), dtype=cdtype())
    for j in range(4):
        psi = jnp.kron(psi, rx(thetas[j]) @ jnp.array([1, 0], dtype=cdtype()))
    return psi


def _rx_on_zero(theta):
    ket0 = jnp.array([1, 0], dtype=cdtype())
    psi = rx(theta) @ ket0
    return jnp.outer(psi, jnp.conjugate(psi))


def encoder_head(rho_in, weights, n_data: int):
    """Append two |0⟩ ancilla, apply V strongly entangling layers, keep 2 qubits.

    First encoder layer: n_data=4 (6-qubit head). Later layers: n_data=2
    (4-qubit head). Output qubits are the first two wires, matching Fig. 2.
    """
    n = n_data + 2
    rho = kron_dm(rho_in, ket0_dm(2))
    rho = strongly_entangling_layers(rho, weights, n)
    return partial_trace(rho, n, keep=(0, 1))


def integration(rho_h1, rho_h2, weights):
    """Combine two 2-qubit head states on 4 qubits, reduce to 2 qubits."""
    rho = kron_dm(rho_h1, rho_h2)
    rho = strongly_entangling_layers(rho, weights, 4)
    return partial_trace(rho, 4, keep=(0, 1))


def connector(rho_encoder, rho_ancilla, ry_angles):
    """CNOT from each encoder qubit to each decoder ancilla, then RY on ancilla.

    Register: [e0, e1, a0, a1]. Returns the 2-qubit ancilla state.
    """
    rho = kron_dm(rho_encoder, rho_ancilla)
    cnot = jnp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=cdtype(),
    )
    for e in (0, 1):
        for a in (2, 3):
            rho = apply_gate(rho, cnot, (e, a), 4)
    rho = apply_gate(rho, ry(ry_angles[0]), (2,), 4)
    rho = apply_gate(rho, ry(ry_angles[1]), (3,), 4)
    return partial_trace(rho, 4, keep=(2, 3))


def decoder_block(rho_ancilla, rho_x, weights):
    """Variational decoder over 2 ancilla + 4 data qubits; keep ancilla."""
    rho = kron_dm(rho_ancilla, rho_x)
    rho = strongly_entangling_layers(rho, weights, 6)
    return partial_trace(rho, 6, keep=(0, 1))


def encoder_state(params, rho_x, config: VQTConfig):
    """Run E encoder blocks; return the 2-qubit encoder density matrix ρ_E."""
    h0 = encoder_head(rho_x, params["enc0_heads"][0], n_data=4)
    h1 = encoder_head(rho_x, params["enc0_heads"][1], n_data=4)
    rho = integration(h0, h1, params["enc0_int"])
    for e in range(config.n_encoder - 1):
        h0 = encoder_head(rho, params["enc_heads"][e, 0], n_data=2)
        h1 = encoder_head(rho, params["enc_heads"][e, 1], n_data=2)
        rho = integration(h0, h1, params["enc_int"][e])
    return rho


@partial(jax.jit, static_argnames=("config",))
def next_token_probs(params, tokens, config: VQTConfig):
    """Autoregressive next-token distribution over the 4-token vocabulary.

    Two-qubit computational-basis probabilities, clamped and renormalized.
    """
    rho_x = encode_context(tokens)
    rho_e = encoder_state(params, rho_x, config)
    rho_a = ket0_dm(2)
    for d in range(config.n_decoder):
        rho_a = connector(rho_e, rho_a, params["dec_ry"][d])
        rho_a = decoder_block(rho_a, rho_x, params["dec_var"][d])
    probs = jnp.real(jnp.diag(rho_a))
    probs = jnp.clip(probs, 1e-12, 1.0)
    return probs / jnp.sum(probs)


def nll_loss(params, tokens, target, config: VQTConfig):
    probs = next_token_probs(params, tokens, config)
    return -jnp.log(probs[target])


def batch_nll(params, contexts, targets, config: VQTConfig):
    def one(ctx, tgt):
        return nll_loss(params, ctx, tgt, config)

    return jnp.mean(jax.vmap(one)(contexts, targets))
