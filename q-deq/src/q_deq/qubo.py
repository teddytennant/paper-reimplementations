"""Q-DEQ local discrete solving: candidate directions and eqs 3–6.

8-bit encoding of each mixing coefficient a_i into [-1, 1]:
    a_i = -1 + (2/255) * unsigned8(x_i)
    offset_i = -1
    B_i = (2/255) * [1, 2, 4, ..., 128]
The paper omits numeric offset/B; this bounded map is our documented choice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

BITS = 8
OFFSET = -1.0
BIT_POWERS = jnp.arange(BITS)  # 0..7
# B_i row: (2/255) * [1, 2, 4, ..., 128]
BIT_WEIGHTS = (2.0 / 255.0) * (2.0 ** BIT_POWERS)


def finite_diff_p(F, z, x, v, eta=1e-3):
    """Eq 3: p ≈ [F(z + η v, x) - F(z, x)] / η - v  = (∂F/∂z - I) v."""
    return (F(z + eta * v, x) - F(z, x)) / eta - v


def directional_p(F, z, x, V, eta=1e-3):
    """Stack eq 3 over columns of V, shape (d, n_dirs) -> P (d, n_dirs)."""

    def one(v):
        return finite_diff_p(F, z, x, v, eta=eta)

    return jax.vmap(one, in_axes=1, out_axes=1)(V)


def gram_h(P, r):
    """Eq 4: G_ij = <p_i, p_j>, h_i = <p_i, r>. Columns of P are p_i."""
    G = P.T @ P
    h = P.T @ r
    return G, h


def quadratic_energy(a, G, h, r):
    """E(a) = 1/2 ||r + P a||^2 = 1/2 a^T G a + h^T a + 1/2 ||r||^2."""
    return 0.5 * a @ G @ a + h @ a + 0.5 * jnp.dot(r, r)


def binary_encoding_matrix(n_dirs, dtype=jnp.float32):
    """Block-diagonal B (n_dirs, n_dirs * BITS) with each block BIT_WEIGHTS."""
    w = BIT_WEIGHTS.astype(dtype)[None, :]
    return jnp.kron(jnp.eye(n_dirs, dtype=dtype), w)


def encoding_offset(n_dirs, dtype=jnp.float32):
    return jnp.full((n_dirs,), OFFSET, dtype=dtype)


def decode_coefficients(x_bits, n_dirs, dtype=None):
    """a_i = -1 + (2/255) * unsigned 8-bit interpretation of x_i."""
    if dtype is None:
        dtype = x_bits.dtype
    bits = x_bits.reshape(n_dirs, BITS).astype(dtype)
    unsigned = bits @ (2.0 ** jnp.arange(BITS, dtype=dtype))
    return OFFSET + (2.0 / 255.0) * unsigned


def encode_qubo(G, h, r=None):
    """Eq 5: a = offset + B x  =>  E(x) = x^T Q x + q^T x + const.

    Q = 1/2 B^T G B
    q = B^T (G offset + h)
    const = 1/2 offset^T G offset + h^T offset + 1/2 ||r||^2
    """
    n_dirs = G.shape[0]
    dtype = G.dtype
    B = binary_encoding_matrix(n_dirs, dtype=dtype)
    offset = encoding_offset(n_dirs, dtype=dtype)
    Q = 0.5 * (B.T @ G @ B)
    q = B.T @ (G @ offset + h)
    const = 0.5 * offset @ G @ offset + h @ offset
    if r is not None:
        const = const + 0.5 * jnp.dot(r, r)
    return Q, q, const, B, offset


def merge_qubo_diagonal(Q, q):
    """Because x_j^2 = x_j, add Q_ii to q_i and zero the diagonal of Q."""
    diag = jnp.diag(Q)
    q_m = q + diag
    Q_m = Q - jnp.diag(diag)
    return Q_m, q_m


def qubo_energy(x, Q, q, const=0.0):
    return x @ Q @ x + q @ x + const


def qubo_to_ising(Q, q):
    """Eq 6: s = 2x - 1, x = (s+1)/2.

    J = (1/4) Q
    b = (1/2)(q + Q 1)
    """
    ones = jnp.ones((Q.shape[0],), dtype=Q.dtype)
    J = 0.25 * Q
    b = 0.5 * (q + Q @ ones)
    const = 0.25 * ones @ Q @ ones + 0.5 * q @ ones
    return J, b, const


def ising_energy(s, J, b, const=0.0):
    return s @ J @ s + b @ s + const


def bits_to_spins(x):
    return 2.0 * x - 1.0


def candidate_directions(z, r, hist, key, n_dirs):
    """Directions from residual, history diffs, and Rademacher padding.

    hist: (H, d) recent states (oldest first). Near-zero diffs are replaced
    with Rademacher vectors so the subspace stays useful.
    Returns V with shape (d, n_dirs).
    """
    d = z.shape[-1]
    dtype = z.dtype
    key_r, key_h, key_e = jax.random.split(key, 3)
    rad_r = jax.random.rademacher(key_r, (d,), dtype=dtype)
    r_norm = jnp.linalg.norm(r)
    v_r = jnp.where(r_norm > 1e-8, r, rad_r)

    H = hist.shape[0]
    deltas = z[None, :] - hist
    dnorms = jnp.linalg.norm(deltas, axis=-1, keepdims=True)
    rad_h = jax.random.rademacher(key_h, hist.shape, dtype=dtype)
    deltas = jnp.where(dnorms > 1e-6, deltas, rad_h)

    extra = jax.random.rademacher(key_e, (n_dirs, d), dtype=dtype)
    stacked = jnp.concatenate([v_r[None, :], deltas, extra], axis=0)
    V = stacked[:n_dirs]
    return V.T
