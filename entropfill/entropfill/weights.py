"""Regularized head weights, equation (4).

Eligible heads are weighted by C_h^{1/tau}. Ineligible heads get zero
until the floor is applied. A group with no eligible head falls back to
the uniform 1/r, which is what keeps the mixture a probability.
"""

import jax.numpy as jnp

from entropfill.pooling import collision, sink_mass


def head_weights(attention, sink, z_min, tau, zeta):
    """Return (G, r) mixture weights and the per-head collision.

    Eligibility is a property of the head, so the sink mass is averaged
    over queries before the threshold. A head that puts almost all of its
    mass on the sink is dropped, which is the isolation the section names.
    """
    z = jnp.mean(sink_mass(attention, sink), axis=-1)
    eligible = z >= z_min
    residual = attention * (1.0 - sink.reshape((1, 1, 1, -1)))
    total = jnp.sum(residual, axis=-1, keepdims=True)
    safe_total = jnp.where(total > 0, total, 1.0)
    conditional = jnp.where(eligible[..., None, None], residual / safe_total, 0.0)
    c = jnp.mean(collision(conditional), axis=-1)
    raw = jnp.where(eligible, c ** (1.0 / tau), 0.0)
    group_mass = jnp.sum(raw, axis=-1, keepdims=True)
    any_eligible = group_mass > 0
    r = attention.shape[1]
    normalized = jnp.where(
        any_eligible, raw / jnp.maximum(group_mass, 1e-30), jnp.full_like(raw, 1.0 / r)
    )
    weights = (1.0 - zeta) * normalized + zeta / r
    return weights, c
