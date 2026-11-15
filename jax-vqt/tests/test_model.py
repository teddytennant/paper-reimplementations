"""Architecture invariants from arXiv:2609.18565: encoding, registers, readout."""

import jax
import jax.numpy as jnp
import pytest

from jax_vqt.circuit import ket0_dm
from jax_vqt.model import (
    VQTConfig,
    batch_nll,
    count_params,
    encode_context,
    encode_statevector,
    encoder_head,
    encoder_state,
    init_params,
    next_token_probs,
    nll_loss,
    parameter_count,
)
from jax_vqt.train import greedy_generate


PAPER_COUNTS = {
    (2, 2, 1): 124,
    (2, 2, 2): 244,
    (2, 2, 3): 364,
    (3, 3, 2): 354,
}


def test_parameter_count_formula_and_table():
    for (e, d, v), n in PAPER_COUNTS.items():
        assert parameter_count(e, d, v) == n
        cfg = VQTConfig(n_encoder=e, n_decoder=d, n_var_layers=v)
        params = init_params(jax.random.PRNGKey(0), cfg)
        assert count_params(params) == n
        assert cfg.name == f"q_e{e}_d{d}_v{v}"


def test_angle_encoding_equation_2():
    tokens = jnp.array([0, 1, 2, 3])
    rho = encode_context(tokens)
    psi = encode_statevector(tokens)
    assert rho.shape == (16, 16)
    assert jnp.allclose(rho, jnp.outer(psi, psi.conj()), atol=1e-10)
    assert jnp.allclose(jnp.trace(rho), 1.0, atol=1e-12)
    # x=0 → RX(0)|0⟩ = |0⟩ on qubit 0, so P(qubit0=0) = 1
    # reshape to (q0,q1,q2,q3, q0',...)
    probs = jnp.real(jnp.diag(rho)).reshape(2, 2, 2, 2)
    p_q0_zero = probs[0].sum()
    assert jnp.allclose(p_q0_zero, 1.0, atol=1e-12)
    # x=2 → RX(π)|0⟩ ∝ |1⟩ on qubit 2
    p_q2_one = probs[:, :, 1, :].sum()
    assert jnp.allclose(p_q2_one, 1.0, atol=1e-12)
    # all-zero context is |0000⟩
    rho0 = encode_context(jnp.array([0, 0, 0, 0]))
    assert jnp.allclose(rho0, ket0_dm(4), atol=1e-12)


def test_register_sizes():
    """Active widths from §3: 4-qubit encode, 6-qubit first head, 4-qubit later
    head/integration, 6-qubit decoder, 2-qubit readout. Max width 6."""
    cfg = VQTConfig(n_encoder=2, n_decoder=2, n_var_layers=1)
    params = init_params(jax.random.PRNGKey(3), cfg)
    rho_x = encode_context(jnp.array([0, 1, 2, 3]))
    assert rho_x.shape == (16, 16)  # 4 qubits
    h = encoder_head(rho_x, params["enc0_heads"][0], n_data=4)
    assert h.shape == (4, 4)  # 2-qubit reduced state
    rho_e = encoder_state(params, rho_x, cfg)
    assert rho_e.shape == (4, 4)
    probs = next_token_probs(params, jnp.array([0, 1, 2, 3]), cfg)
    assert probs.shape == (4,)  # 2-qubit computational basis → |V|=4


def test_readout_is_probability_simplex():
    cfg = VQTConfig()
    params = init_params(jax.random.PRNGKey(4), cfg)
    for tokens in ([0, 0, 0, 0], [0, 1, 2, 3], [3, 3, 3, 3], [2, 0, 1, 3]):
        p = next_token_probs(params, jnp.array(tokens), cfg)
        assert p.shape == (cfg.vocab_size,)
        assert jnp.all(p >= 0)
        assert jnp.allclose(jnp.sum(p), 1.0, atol=1e-10)


def test_next_token_interface():
    cfg = VQTConfig(n_encoder=2, n_decoder=1, n_var_layers=1)
    params = init_params(jax.random.PRNGKey(5), cfg)
    ctx = jnp.array([0, 0, 0, 3], dtype=jnp.int32)  # "AAA "
    p = next_token_probs(params, ctx, cfg)
    assert p.shape == (4,)
    loss = nll_loss(params, ctx, jnp.array(1), cfg)  # next should be B=1 in det grammar
    assert jnp.isfinite(loss)
    contexts = jnp.stack([ctx, jnp.array([0, 0, 3, 1])])
    targets = jnp.array([1, 1])
    bl = batch_nll(params, contexts, targets, cfg)
    assert jnp.isfinite(bl)


def test_gradients_finite():
    cfg = VQTConfig(n_encoder=2, n_decoder=1, n_var_layers=1)
    params = init_params(jax.random.PRNGKey(6), cfg)
    ctx = jnp.array([0, 0, 0, 3])
    tgt = jnp.array(1)

    def loss_fn(p):
        return nll_loss(p, ctx, tgt, cfg)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert jnp.isfinite(loss)
    for g in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(g))


def test_greedy_generation_length():
    cfg = VQTConfig(n_encoder=1, n_decoder=1, n_var_layers=1)
    params = init_params(jax.random.PRNGKey(7), cfg)
    ctx = jnp.array([0, 0, 0, 3], dtype=jnp.int32)
    out = greedy_generate(params, ctx, 6, cfg)
    assert out.shape == (6,)
    assert jnp.all((out >= 0) & (out < 4))


@pytest.mark.parametrize("e,d,v", [(2, 2, 1), (1, 1, 1)])
def test_init_matches_formula(e, d, v):
    cfg = VQTConfig(n_encoder=e, n_decoder=d, n_var_layers=v)
    params = init_params(jax.random.PRNGKey(8), cfg)
    assert count_params(params) == parameter_count(e, d, v)
