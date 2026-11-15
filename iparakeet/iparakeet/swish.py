"""Minimax Swish approximation from arXiv 2609.30846.

Coefficients are the paper's a = -0.1240 and c = 2.4632. Do not retune them
to chase the reported 0.039 max-abs error; the formula is

    sw(x) = x * sigmoid(x)
    tanh_hat(u) = sign(u) * (a * (min(|u|, c) - c)^2 + 1)
    sw_hat(x) = x * (1 + tanh_hat(x/2)) / 2
"""

from __future__ import annotations

import jax.numpy as jnp

SWISH_A = -0.1240
SWISH_C = 2.4632


def swish(x):
    """``x * sigmoid(x)``."""
    x = jnp.asarray(x)
    return x * jax_sigmoid(x)


def jax_sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def tanh_hat(u, a=SWISH_A, c=SWISH_C):
    """Odd saturating stand-in for tanh. Clips the quadratic at ``|u| = c``."""
    u = jnp.asarray(u)
    mag = jnp.minimum(jnp.abs(u), c)
    core = a * (mag - c) ** 2 + 1.0
    return jnp.sign(u) * core


def swish_hat(x, a=SWISH_A, c=SWISH_C):
    """``x * (1 + tanh_hat(x/2)) / 2``."""
    x = jnp.asarray(x)
    return x * (1.0 + tanh_hat(x / 2.0, a=a, c=c)) / 2.0


def hard_swish(x):
    """Hard-Swish ``x * relu6(x + 3) / 6``."""
    x = jnp.asarray(x)
    return x * jnp.clip(x + 3.0, 0.0, 6.0) / 6.0
