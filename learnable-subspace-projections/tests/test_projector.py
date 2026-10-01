import jax
import jax.numpy as jnp

from learnable_subspace import (
    dense_projector,
    forward_input,
    forward_output,
    orthogonal_factor,
    orthogonality_penalty,
    project_activations,
    sample_direction_mask,
)


def test_orthonormal_projector_and_activation_forward():
    key = jax.random.PRNGKey(0)
    kW, kx, kV = jax.random.split(key, 3)
    din, dout, k = 8, 6, 3
    W = jax.random.normal(kW, (dout, din))
    b = jax.random.normal(kW, (dout,))
    x = jax.random.normal(kx, (din,))
    V = jax.random.normal(kV, (din, k))
    U = orthogonal_factor(V)

    eye_k = jnp.eye(k)
    assert jnp.allclose(U.T @ U, eye_k, atol=1e-5, rtol=1e-5)

    P = dense_projector(U, alpha=1.0, mask=None)
    assert jnp.allclose(P, P.T, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(P @ P, P, atol=1e-5, rtol=1e-5)

    y = forward_input(W, x, b, U, alpha=1.0, mask=jnp.ones((k,)))
    y_ref = (W @ P) @ x + b
    assert jnp.allclose(y, y_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(project_activations(x, U), P @ x, atol=1e-5, rtol=1e-5)

    rank = int(jnp.linalg.matrix_rank(W @ P, tol=1e-8))
    assert rank <= din - k
    assert int(jnp.linalg.matrix_rank(P, tol=1e-8)) == din - k


def test_output_side_forward_matches_dense_projector():
    key = jax.random.PRNGKey(1)
    kW, kx, kV = jax.random.split(key, 3)
    din, dout, k = 5, 7, 2
    W = jax.random.normal(kW, (dout, din))
    b = jax.random.normal(kx, (dout,))
    x = jax.random.normal(kV, (din,))
    U = orthogonal_factor(jax.random.normal(key, (dout, k)))
    P = dense_projector(U)
    y = forward_output(W, x, b, U, alpha=1.0, mask=jnp.ones((k,)))
    y_ref = P @ (W @ x) + b
    assert jnp.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_bias_is_added_unprojected():
    din, dout, k = 4, 3, 1
    W = jnp.eye(dout, din)
    U = jnp.zeros((din, k)).at[:, 0].set(1.0)
    x = jnp.ones((din,))
    b = jnp.arange(dout, dtype=jnp.float64)
    y = forward_input(W, x, b, U, alpha=1.0)
    y0 = forward_input(W, x, jnp.zeros_like(b), U, alpha=1.0)
    assert jnp.allclose(y - y0, b, atol=1e-5, rtol=1e-5)


def test_direction_dropout_is_unscaled_bernoulli():
    mask = sample_direction_mask(jax.random.PRNGKey(2), k=64, dropout_p=0.25)
    assert mask.shape == (64,)
    assert jnp.all((mask == 0.0) | (mask == 1.0))
    # A 1/(1-p) rescale would put kept entries at 1/0.75, not at 1.
    assert float(jnp.max(mask)) == 1.0

    din, k = 4, 3
    U = jnp.eye(din, k)
    x = jnp.arange(din, dtype=jnp.float64)
    m = jnp.array([1.0, 0.0, 1.0])
    alpha = 0.5
    got = project_activations(x, U, alpha=alpha, mask=m)
    P = dense_projector(U, alpha=alpha, mask=m)
    assert jnp.allclose(got, P @ x, atol=1e-5, rtol=1e-5)


def test_alpha_zero_is_dense_forward():
    key = jax.random.PRNGKey(3)
    W = jax.random.normal(key, (4, 5))
    b = jax.random.normal(key, (4,))
    x = jax.random.normal(key, (5,))
    U = orthogonal_factor(jax.random.normal(key, (5, 2)))
    y = forward_input(W, x, b, U, alpha=0.0)
    assert jnp.allclose(y, W @ x + b, atol=1e-5, rtol=1e-5)


def test_orthogonality_penalty_raw_columns():
    V_orth = jnp.eye(6, 3)
    assert jnp.allclose(orthogonality_penalty(V_orth), 0.0, atol=1e-8)

    # Identical raw columns: each inner product is 6, six ordered pairs, k=3.
    V_same = jnp.stack([jnp.ones((6,)), jnp.ones((6,)), jnp.ones((6,))], axis=1)
    penalty_same = orthogonality_penalty(V_same)
    assert float(penalty_same) > 0.0
    assert jnp.allclose(penalty_same, 6.0, atol=1e-8)

    # Anti-aligned raw columns give a negative penalty. Not a cosine.
    V_anti = jnp.array([[1.0, -1.0], [0.0, 0.0]])
    assert float(orthogonality_penalty(V_anti)) < 0.0

    V_single = jnp.ones((5, 1))
    assert float(orthogonality_penalty(V_single)) == 0.0
