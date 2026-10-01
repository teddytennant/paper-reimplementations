import jax
import jax.numpy as jnp

from learnable_subspace import (
    allocate,
    cholesky_covariance,
    dense_projector,
    orthogonal_factor,
    output_side_init,
    reconstruction_energy,
    savings,
    tail_energy,
    untied_c,
    untied_unit,
    whitened_input_init,
)


def test_isotropic_whitened_error_matches_tail_energy():
    din, dout, r = 5, 7, 2
    W = jax.random.normal(jax.random.PRNGKey(0), (dout, din))
    c = 2.5
    n = din
    X = jnp.sqrt(c * n) * jnp.eye(din)
    init = whitened_input_init(W, X, r)

    energy = reconstruction_energy(W, W @ init.P_init, init.S)
    tail = tail_energy(init.sigma, r)
    rel = jnp.abs(energy - tail) / tail
    assert float(rel) < 1e-4

    U = orthogonal_factor(init.V)
    P = dense_projector(U)
    assert jnp.allclose(P, init.P_init, atol=1e-5, rtol=1e-5)
    assert init.V.shape == (din, din - r)


def test_nonisotropic_projector_preserves_kept_subspace():
    kW, kX = jax.random.split(jax.random.PRNGKey(1))
    din, dout, r = 6, 5, 3
    W = jax.random.normal(kW, (dout, din))
    X = jax.random.normal(kX, (30, din)).at[:, 0].mul(5.0)
    init = whitened_input_init(W, X, r)
    lhs = W @ init.P_init @ init.C_r
    rhs = W @ init.C_r
    assert jnp.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)

    U = orthogonal_factor(init.V)
    assert jnp.allclose(dense_projector(U), init.P_init, atol=1e-5, rtol=1e-5)


def test_ridge_when_cholesky_is_nonfinite():
    X = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    S = cholesky_covariance(X)
    assert bool(jnp.all(jnp.isfinite(S)))
    G = S @ S.T
    assert jnp.all(jnp.linalg.eigvalsh(G) > 0.0)

    W = jax.random.normal(jax.random.PRNGKey(4), (4, 3))
    init = whitened_input_init(W, X, r=1)
    assert bool(jnp.all(jnp.isfinite(init.V)))
    assert bool(jnp.all(jnp.isfinite(init.P_init)))


def test_output_side_uses_trailing_left_singular_vectors():
    W = jax.random.normal(jax.random.PRNGKey(2), (6, 4))
    k = 2
    V = output_side_init(W, k)
    U_w, _sigma, _Vh = jnp.linalg.svd(W, full_matrices=True)
    trailing = U_w[:, -k:]
    U = orthogonal_factor(V)
    assert jnp.allclose(U @ U.T, trailing @ trailing.T, atol=1e-5, rtol=1e-5)
    assert V.shape == (W.shape[0], k)


def test_untied_savings_formula():
    assert untied_c(4, 6) == 10
    unit = untied_unit(4, 6, delta=[0, 0, 0], side="input")
    assert unit["c"] == 10
    assert unit["d"] == 4
    assert unit["N"] == 24
    assert savings(0, unit["N"], unit["c"], unit["d"]) == 0.0
    assert savings(2, unit["N"], unit["c"], unit["d"]) == 24 - 10 * (4 - 2)
    output = untied_unit(4, 6, delta=[0, 1], side="output")
    assert output["d"] == 6


def test_allocator_compresses_cheap_unit_first():
    # Untied input-side layers, din = dout = 4.
    # c = 8, N = 16, d = 4.
    # s(0) = 0, s(1) = -8, s(2) = 0, s(3) = 8, s(4) = 16.
    # A pays KL 1 to reach savings 8. B pays KL 100 for that same move.
    # Target 8. The greedy jump must land on (k_A, k_B) = (3, 0).
    delta_a = [0, 0, 0, 1, 2]
    delta_b = [0, 0, 0, 100, 200]
    units = [
        untied_unit(4, 4, delta_a, side="input"),
        untied_unit(4, 4, delta_b, side="input"),
    ]
    chosen = allocate(units, target_savings=8)
    assert chosen == (3, 0)
