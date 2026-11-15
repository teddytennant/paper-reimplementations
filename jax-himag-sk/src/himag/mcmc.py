"""Short MCMC runner for the Algorithm 1 down-up walk."""

from __future__ import annotations

import jax

from himag.walk import down_up_step

Array = jax.Array


def run_chain(key: Array, x0: Array, J: Array, h, beta: float, n_steps: int) -> Array:
    """Start at a valid slice vector ``x0`` and take ``n_steps`` down-up steps.

    Returns samples of shape ``(n_steps, d)``, one after each step (x0 omitted).
    """

    def body(carry, _):
        key, x = carry
        key, sub = jax.random.split(key)
        x = down_up_step(sub, x, J, h, beta)
        return (key, x), x

    _, xs = jax.lax.scan(body, (key, x0), xs=None, length=int(n_steps))
    return xs
