import jax
import jax.numpy as jnp

from learnable_subspace import (
    dense_projector,
    merge_input,
    merge_output,
    merge_tied_input,
    orthogonal_complement,
    orthogonal_factor,
)


def test_input_merge_matches_weight_times_projector():
    key = jax.random.PRNGKey(0)
    kW, kU = jax.random.split(key)
    din, dout, k = 8, 5, 3
    W = jax.random.normal(kW, (dout, din))
    U = orthogonal_factor(jax.random.normal(kU, (din, k)))
    A, B = merge_input(W, U)
    r = din - k
    assert A.shape == (r, din)
    assert B.shape == (dout, r)
    P = dense_projector(U)
    err = jnp.max(jnp.abs(W @ P - B @ A))
    assert float(err) < 1e-5


def test_output_merge_matches_projector_times_weight():
    key = jax.random.PRNGKey(1)
    kW, kU = jax.random.split(key)
    din, dout, k = 5, 7, 2
    W = jax.random.normal(kW, (dout, din))
    U = orthogonal_factor(jax.random.normal(kU, (dout, k)))
    U_perp = orthogonal_complement(U)
    P = dense_projector(U)
    recon = U_perp @ (U_perp.T @ W)
    err = jnp.max(jnp.abs(P @ W - recon))
    assert float(err) < 1e-5
    A, B = merge_output(W, U)
    assert float(jnp.max(jnp.abs(P @ W - B @ A))) < 1e-5


def test_tied_pair_shares_one_A():
    key = jax.random.PRNGKey(2)
    k1, k2, kU = jax.random.split(key, 3)
    din, k = 8, 3
    W1 = jax.random.normal(k1, (5, din))
    W2 = jax.random.normal(k2, (7, din))
    U = orthogonal_factor(jax.random.normal(kU, (din, k)))
    A, Bs = merge_tied_input([W1, W2], U)
    A1, B1 = merge_input(W1, U)
    A2, B2 = merge_input(W2, U)
    assert jnp.allclose(A, A1, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(A, A2, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(Bs[0], B1, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(Bs[1], B2, atol=1e-5, rtol=1e-5)

    P = dense_projector(U)
    err1 = jnp.max(jnp.abs(W1 @ P - Bs[0] @ A))
    err2 = jnp.max(jnp.abs(W2 @ P - Bs[1] @ A))
    assert float(err1) < 1e-5
    assert float(err2) < 1e-5
    # One shared A, not two independent factors.
    assert A is not None and Bs[0].shape[1] == A.shape[0] == Bs[1].shape[1]
