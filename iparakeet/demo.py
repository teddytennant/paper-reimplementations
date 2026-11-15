"""CPU demo: Swish error, eq. 10 vs eq. 11, and a tiny attention compare."""

import jax
import jax.numpy as jnp

from iparakeet.attention import (
    activation_scales,
    float_attention,
    fuse_scores_dyadic,
    fuse_scores_reference,
    integer_attention,
)
from iparakeet.swish import swish, swish_hat


def main():
    xs = jnp.linspace(-8.0, 8.0, 20001)
    swish_err = jnp.max(jnp.abs(swish_hat(xs) - swish(xs)))

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    qc = jax.random.randint(k1, (4, 4), -40, 41, dtype=jnp.int8)
    qp = jax.random.randint(k2, (4, 7), -40, 41, dtype=jnp.int8)
    ref = fuse_scores_reference(qc, qp, 0.05, 0.03, 0.02, 4)
    dy = fuse_scores_dyadic(qc, qp, 0.05, 0.03, 0.02, 4, n=16)
    fusion_gap = jnp.max(jnp.abs(ref - dy.astype(ref.dtype)))

    keys = jax.random.split(jax.random.PRNGKey(4), 6)
    length, dk, dv = 4, 8, 4
    Q = jax.random.normal(keys[0], (length, dk)) * 0.5
    K = jax.random.normal(keys[1], (length, dk)) * 0.5
    V = jax.random.normal(keys[2], (length, dv))
    P = jax.random.normal(keys[3], (2 * length - 1, dk)) * 0.5
    u = jax.random.normal(keys[4], (dk,)) * 0.1
    v = jax.random.normal(keys[5], (dk,)) * 0.1
    scales = activation_scales(Q, K, V, P, u, v)
    attn_err = jnp.max(
        jnp.abs(
            integer_attention(Q, K, V, P, u, v, scales)
            - float_attention(Q, K, V, P, u, v)
        )
    )

    print(f"max swish abs error: {float(swish_err):.6f}")
    print(f"max |eq10 - eq11|: {float(fusion_gap):.6f}")
    print(f"tiny attention max abs error: {float(attn_err):.6f}")


if __name__ == "__main__":
    main()
