"""Robust losses l_σ(u) = G(u²/σ²) from §2, Examples 1–4."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

LossName = Literal["huber", "fair", "cauchy", "welsch"]
LOSSES: tuple[LossName, ...] = ("huber", "fair", "cauchy", "welsch")

# G'_+(0) = 1/2 and C_G = sup |G'| = 1/2 for every example in the paper.
G_PRIME_PLUS_0 = 0.5
C_G = 0.5


def G(s, loss: LossName = "welsch"):
    """Windowing function G in l_σ(u) = G(u²/σ²)."""
    s = jnp.asarray(s)
    if loss == "huber":
        # Example 1: G(s) = s/2 on s ≤ 1, √s − 1/2 on s > 1.
        return jnp.where(s <= 1.0, 0.5 * s, jnp.sqrt(jnp.maximum(s, 0.0)) - 0.5)
    if loss == "fair":
        # Example 2: G(s) = √s − log(1 + √s).
        root = jnp.sqrt(jnp.maximum(s, 0.0))
        return root - jnp.log1p(root)
    if loss == "cauchy":
        # Example 3: G(s) = log(1 + s/2).
        return jnp.log1p(0.5 * s)
    if loss == "welsch":
        # Example 4: G(s) = 1 − exp(−s/2).
        return 1.0 - jnp.exp(-0.5 * s)
    raise ValueError(f"unknown loss {loss!r}")


def G_prime(s, loss: LossName = "welsch"):
    """G'(s) used in the kernel GD update (4)."""
    s = jnp.asarray(s)
    if loss == "huber":
        # 1/2 on s ≤ 1, 1/(2√s) on s > 1; right-derivative at 0 is 1/2.
        return jnp.where(s <= 1.0, 0.5, 0.5 / jnp.sqrt(jnp.maximum(s, 1e-12)))
    if loss == "fair":
        # d/ds [√s − log(1+√s)] = 1 / (2(1+√s)), equal to 1/2 at s = 0.
        return 0.5 / (1.0 + jnp.sqrt(jnp.maximum(s, 0.0)))
    if loss == "cauchy":
        # d/ds log(1 + s/2) = 1/(2 + s).
        return 1.0 / (2.0 + s)
    if loss == "welsch":
        # d/ds [1 − exp(−s/2)] = (1/2) exp(−s/2).
        return 0.5 * jnp.exp(-0.5 * s)
    raise ValueError(f"unknown loss {loss!r}")


def G_prime_plus_0(loss: LossName = "welsch") -> float:
    """Right derivative G'_+(0) in (1). Equal to 1/2 for Examples 1–4."""
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}")
    return G_PRIME_PLUS_0


def l_sigma(u, sigma, loss: LossName = "welsch"):
    """Robust loss l_σ(u) = G(u²/σ²)."""
    u = jnp.asarray(u)
    sigma = jnp.asarray(sigma)
    return G((u / sigma) ** 2, loss=loss)


def influence_weight(residual, sigma, loss: LossName = "welsch"):
    """Pointwise factor G'(ξ)(f(x)−y) in (4), ξ = (f(x)−y)²/σ²."""
    residual = jnp.asarray(residual)
    xi = (residual / sigma) ** 2
    return G_prime(xi, loss=loss) * residual
