"""Lock eqs 3–6: finite-diff p, quadratic energy, 8-bit QUBO, Ising."""

import jax
import jax.numpy as jnp
import numpy as np

from q_deq.qubo import (
    BITS,
    binary_encoding_matrix,
    bits_to_spins,
    decode_coefficients,
    directional_p,
    encode_qubo,
    encoding_offset,
    finite_diff_p,
    gram_h,
    ising_energy,
    merge_qubo_diagonal,
    quadratic_energy,
    qubo_energy,
    qubo_to_ising,
)


def _linear_F(A, B):
    def F(z, x):
        return A @ z + B @ x

    return F


def test_eq3_finite_diff_matches_autodiff_linear():
    key = jax.random.PRNGKey(1)
    d, din = 8, 5
    k1, k2, k3, k4 = jax.random.split(key, 4)
    A = 0.4 * jax.random.normal(k1, (d, d))
    B = 0.2 * jax.random.normal(k2, (d, din))
    z = jax.random.normal(k3, (d,))
    x = jax.random.normal(k4, (din,))
    v = jax.random.normal(k1, (d,))
    F = _linear_F(A, B)

    p_fd = finite_diff_p(F, z, x, v, eta=1e-2)
    _, Jv = jax.jvp(lambda zz: F(zz, x), (z,), (v,))
    p_ad = Jv - v
    # Linear F: finite difference is exact for any eta.
    assert jnp.allclose(p_fd, p_ad, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(p_fd, (A - jnp.eye(d)) @ v, atol=1e-5)


def test_eq3_finite_diff_matches_jvp_nonlinear():
    key = jax.random.PRNGKey(2)
    d = 6
    k1, k2, k3 = jax.random.split(key, 3)
    W = 0.3 * jax.random.normal(k1, (d, d))
    z = jax.random.normal(k2, (d,))
    x = jax.random.normal(k3, (d,))
    v = jax.random.normal(k1, (d,))

    def F(zz, xx):
        return jnp.tanh(W @ zz + 0.2 * xx)

    p_fd = finite_diff_p(F, z, x, v, eta=1e-3)
    _, Jv = jax.jvp(lambda zz: F(zz, x), (z,), (v,))
    p_ad = Jv - v
    assert jnp.allclose(p_fd, p_ad, atol=2e-3, rtol=2e-3)


def test_eq3_stacked_directions():
    key = jax.random.PRNGKey(3)
    d, n = 7, 4
    A = 0.5 * jnp.eye(d)
    B = 0.1 * jnp.ones((d, d))
    F = _linear_F(A, B)
    z = jax.random.normal(key, (d,))
    x = jax.random.normal(key, (d,))
    V = jax.random.normal(key, (d, n))
    P = directional_p(F, z, x, V, eta=1e-2)

    def p_ad(v):
        _, Jv = jax.jvp(lambda zz: F(zz, x), (z,), (v,))
        return Jv - v

    P_ad = jax.vmap(p_ad, in_axes=1, out_axes=1)(V)
    assert jnp.allclose(P, P_ad, atol=1e-5)


def test_eq4_G_h_match_quadratic_residual_energy():
    key = jax.random.PRNGKey(4)
    d, n = 10, 5
    k1, k2, k3 = jax.random.split(key, 3)
    P = jax.random.normal(k1, (d, n))
    r = jax.random.normal(k2, (d,))
    a = jax.random.normal(k3, (n,))
    G, h = gram_h(P, r)
    E = quadratic_energy(a, G, h, r)
    E_direct = 0.5 * jnp.sum((r + P @ a) ** 2)
    assert jnp.allclose(E, E_direct, atol=1e-5)
    # G, h reconstruction
    assert jnp.allclose(G, P.T @ P, atol=1e-5)
    assert jnp.allclose(h, P.T @ r, atol=1e-5)


def test_eq5_eight_bit_encoding_range():
    n = 3
    B = binary_encoding_matrix(n)
    offset = encoding_offset(n)
    assert B.shape == (n, n * BITS)
    zeros = jnp.zeros((n * BITS,))
    ones = jnp.ones((n * BITS,))
    a0 = offset + B @ zeros
    a1 = offset + B @ ones
    assert jnp.allclose(a0, jnp.full((n,), -1.0))
    assert jnp.allclose(a1, jnp.full((n,), 1.0))
    # mid: unsigned 128 -> -1 + (2/255)*128
    bits = jnp.zeros((n, BITS)).at[:, 7].set(1.0).reshape(-1)
    a = offset + B @ bits
    assert jnp.allclose(
        a, jnp.full((n,), -1.0 + (2.0 / 255.0) * 128.0), atol=1e-6, rtol=1e-5
    )


def test_eq5_Q_q_match_substitution_and_diagonal_merge():
    key = jax.random.PRNGKey(5)
    d, n = 9, 3
    k1, k2, k3 = jax.random.split(key, 3)
    P = jax.random.normal(k1, (d, n))
    r = jax.random.normal(k2, (d,))
    G, h = gram_h(P, r)
    Q, q, const, B, offset = encode_qubo(G, h, r)

    rng = np.random.default_rng(0)
    for _ in range(8):
        x = jnp.array(rng.integers(0, 2, size=(n * BITS,)), dtype=jnp.float32)
        a = offset + B @ x
        E4 = quadratic_energy(a, G, h, r)
        E5 = qubo_energy(x, Q, q, const)
        assert jnp.allclose(E4, E5, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(a, decode_coefficients(x, n), atol=1e-6)

    Qm, qm = merge_qubo_diagonal(Q, q)
    assert jnp.allclose(jnp.diag(Qm), 0.0, atol=1e-7)
    for _ in range(8):
        x = jnp.array(rng.integers(0, 2, size=(n * BITS,)), dtype=jnp.float32)
        e_before = qubo_energy(x, Q, q)
        e_after = qubo_energy(x, Qm, qm)
        assert jnp.allclose(e_before, e_after, atol=1e-5)
        # Off the {0,1} cube the merge need not preserve energy; on it, it must.


def test_eq6_ising_energy_agrees_up_to_constant():
    key = jax.random.PRNGKey(6)
    d, n = 8, 2
    k1, k2 = jax.random.split(key, 2)
    P = jax.random.normal(k1, (d, n))
    r = jax.random.normal(k2, (d,))
    G, h = gram_h(P, r)
    Q, q, const, _B, _off = encode_qubo(G, h, r)
    Q, q = merge_qubo_diagonal(Q, q)
    J, b, _c = qubo_to_ising(Q, q)

    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(12):
        x = jnp.array(rng.integers(0, 2, size=(n * BITS,)), dtype=jnp.float32)
        s = bits_to_spins(x)
        E = qubo_energy(x, Q, q, const)
        H = ising_energy(s, J, b)
        diffs.append(np.asarray(H - E))
    diffs = np.array(diffs)
    assert np.allclose(diffs, diffs[0], atol=1e-4)
