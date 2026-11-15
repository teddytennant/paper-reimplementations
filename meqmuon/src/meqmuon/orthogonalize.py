"""Newton-Schulz orthogonalization used by MeqMuon."""

import jax
import jax.numpy as jnp


def newton_schulz(h, steps=5, coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Orthogonalize a matrix with the Muon quintic Newton-Schulz iteration.

    Tall inputs are transposed so the Gram matrix is the smaller side, iterated,
    then transposed back. An exact-zero matrix returns zeros. Otherwise the
    input is scaled by 1 / (||H||_F + eps).
    """
    h = jnp.asarray(h)
    a, b, c = coefficients
    a = jnp.asarray(a, dtype=h.dtype)
    b = jnp.asarray(b, dtype=h.dtype)
    c = jnp.asarray(c, dtype=h.dtype)
    eps = jnp.asarray(eps, dtype=h.dtype)

    fro = jnp.linalg.norm(h, ord="fro")
    # Never divide by zero. Exact zeros stay zeros; every other matrix uses fro + eps.
    safe = jnp.where(fro == 0, jnp.ones_like(fro), fro + eps)
    x = jnp.where(fro == 0, jnp.zeros_like(h), h / safe)

    # Muon transposes when there are more rows than columns so A = X @ X.T is small.
    transposed = h.shape[0] > h.shape[1]
    if transposed:
        x = x.T

    def body(_, state):
        gram = state @ state.T
        return a * state + (b * gram + c * (gram @ gram)) @ state

    steps = jnp.asarray(steps)
    x = jax.lax.fori_loop(jnp.array(0, dtype=steps.dtype), steps, body, x)

    if transposed:
        x = x.T
    return x
