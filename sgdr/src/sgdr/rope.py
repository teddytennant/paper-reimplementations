"""Real-pair rotary position embeddings.

Head dimension d is even. Frequency f = 0 .. d/2 - 1 uses

    omega_f = 1 / (10000 ** (2f / d))

A real vector becomes complex channels x_f = x[2f] + i x[2f+1].
Post-RoPE at absolute position p is x_f * exp(i * omega_f * p).
"""

import jax.numpy as jnp


def rope_frequencies(dim, base=10000.0):
    """Return omega_f for f = 0 .. dim/2 - 1, float64."""
    if dim % 2 != 0:
        raise ValueError(f"head dim must be even, got {dim}")
    half = dim // 2
    freq_index = jnp.arange(half, dtype=jnp.float64)
    return 1.0 / (jnp.asarray(base, dtype=jnp.float64) ** (2.0 * freq_index / dim))


def to_complex(x):
    """Pack real pairs into complex channels. x[..., d] -> [..., d/2]."""
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {x.shape[-1]}")
    return x[..., 0::2] + 1j * x[..., 1::2]


def to_real(z):
    """Inverse of to_complex. z[..., F] complex -> [..., 2F] real."""
    z = jnp.asarray(z)
    paired = jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)
    return paired.reshape(z.shape[:-1] + (z.shape[-1] * 2,))


def apply_rope(x_complex, positions, omegas):
    """Rotate complex channels by exp(i * omega_f * position).

    x_complex: (..., L, F)
    positions: (L,) absolute positions
    omegas: (F,)
    """
    positions = jnp.asarray(positions, dtype=jnp.float64)
    omegas = jnp.asarray(omegas, dtype=jnp.float64)
    phase = positions[..., None] * omegas
    return x_complex * jnp.exp(1j * phase)


def token_complex_product(q_post, k_post):
    """Re(q_post * conj(k_post)) summed over frequency, no 1/sqrt(d).

    q_post: (..., F), k_post: (..., F) broadcastable.
    """
    return jnp.real(q_post * jnp.conj(k_post)).sum(axis=-1)
