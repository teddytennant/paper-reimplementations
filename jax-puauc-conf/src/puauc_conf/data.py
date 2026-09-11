"""Synthetic biased PU data with confidence (one-sample / SAR)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class PUData(NamedTuple):
    x_labeled: jax.Array
    r: jax.Array
    x_unlabeled: jax.Array
    y_unlabeled: jax.Array
    x_all: jax.Array
    y_all: jax.Array
    o: jax.Array
    eta: jax.Array
    e: jax.Array
    pi: jax.Array
    alpha: jax.Array
    c: jax.Array


def true_posterior(
    x: jax.Array,
    mu_p: jax.Array,
    mu_n: jax.Array,
    pi: float,
    var: float = 1.0,
) -> jax.Array:
    """p(y=1|x) for two Gaussians with equal isotropic covariance."""
    log_pp = -0.5 / var * jnp.sum((x - mu_p) ** 2, axis=-1)
    log_pn = -0.5 / var * jnp.sum((x - mu_n) ** 2, axis=-1)
    logits = jnp.log(pi / (1.0 - pi)) + log_pp - log_pn
    return jax.nn.sigmoid(logits)


def propensity_high_density(x: jax.Array, mu_p: jax.Array) -> jax.Array:
    """SAR propensity: positives near μ_p are much more likely to be labeled."""
    dist = jnp.linalg.norm(x - mu_p, axis=-1)
    # High in a ball around μ_p, low outside; strictly positive (positivity).
    return 0.05 + 0.85 * jax.nn.sigmoid(1.5 - 2.0 * dist)


def generate_biased_pu(
    key: jax.Array,
    n: int = 400,
    pi: float = 0.3,
    mu_p: tuple[float, float] = (1.5, 1.5),
    mu_n: tuple[float, float] = (-1.5, -1.5),
    var: float = 1.0,
    confidence: str = "true",
) -> PUData:
    """One-sample SAR draw: mixture → label positives with e(x) → rest unlabeled.

    ``confidence`` is ``\"true\"`` for r(x)=η(x), or ``\"sq\"`` for r=η²
    (strictly increasing on [0, 1]).
    """
    mu_p_a = jnp.asarray(mu_p)
    mu_n_a = jnp.asarray(mu_n)
    k_y, k_xp, k_xn, k_o = jax.random.split(key, 4)
    y = jax.random.bernoulli(k_y, pi, (n,)).astype(jnp.int32)
    xp = mu_p_a + jnp.sqrt(var) * jax.random.normal(k_xp, (n, 2))
    xn = mu_n_a + jnp.sqrt(var) * jax.random.normal(k_xn, (n, 2))
    x = jnp.where(y[:, None] == 1, xp, xn)
    eta = true_posterior(x, mu_p_a, mu_n_a, pi, var=var)
    e = propensity_high_density(x, mu_p_a)
    o_pos = jax.random.bernoulli(k_o, e).astype(jnp.int32)
    o = jnp.where(y == 1, o_pos, jnp.zeros_like(o_pos))
    if confidence == "sq":
        r_all = eta**2
    elif confidence == "true":
        r_all = eta
    else:
        raise ValueError(f"unknown confidence={confidence!r}")
    labeled = o == 1
    unlabeled = o == 0
    n_p = labeled.sum()
    n_u = unlabeled.sum()
    alpha = n_p / (n_p + n_u)
    n_pos = (y == 1).sum()
    c = jnp.where(n_pos > 0, n_p / n_pos, 0.0)
    return PUData(
        x_labeled=x[labeled],
        r=r_all[labeled],
        x_unlabeled=x[unlabeled],
        y_unlabeled=y[unlabeled],
        x_all=x,
        y_all=y,
        o=o,
        eta=eta,
        e=e,
        pi=jnp.asarray(pi),
        alpha=alpha,
        c=c,
    )


def generate_labeled_test(
    key: jax.Array,
    n: int = 400,
    pi: float = 0.3,
    mu_p: tuple[float, float] = (1.5, 1.5),
    mu_n: tuple[float, float] = (-1.5, -1.5),
    var: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Fully labeled i.i.d. test sample from the same mixture."""
    mu_p_a = jnp.asarray(mu_p)
    mu_n_a = jnp.asarray(mu_n)
    k_y, k_xp, k_xn = jax.random.split(key, 3)
    y = jax.random.bernoulli(k_y, pi, (n,)).astype(jnp.int32)
    xp = mu_p_a + jnp.sqrt(var) * jax.random.normal(k_xp, (n, 2))
    xn = mu_n_a + jnp.sqrt(var) * jax.random.normal(k_xn, (n, 2))
    x = jnp.where(y[:, None] == 1, xp, xn)
    return x, y


def discrete_population() -> dict[str, jax.Array]:
    """Tiny finite population with repeated x so 0 < e(x) ≤ 1 (positivity).

    Three atoms. Counts (n_neg, n_pos, n_labeled_pos):
      x=0: (8, 4, 2)
      x=1: (6, 6, 3)
      x=2: (2, 10, 5)
    """
    atoms = [
        (0.0, 8, 4, 2),
        (1.0, 6, 6, 3),
        (2.0, 2, 10, 5),
    ]
    xs, ys, os = [], [], []
    for x, n_neg, n_pos, n_lab in atoms:
        xs.extend([x] * (n_neg + n_pos))
        ys.extend([0] * n_neg + [1] * n_pos)
        os.extend([0] * n_neg + [1] * n_lab + [0] * (n_pos - n_lab))
    x = jnp.asarray(xs)
    y = jnp.asarray(ys, dtype=jnp.int32)
    o = jnp.asarray(os, dtype=jnp.int32)
    eta = jnp.zeros_like(x)
    u = jnp.zeros_like(x)
    for xv in (0.0, 1.0, 2.0):
        m = x == xv
        eta = jnp.where(m, y[m].mean(), eta)
        u = jnp.where(m, o[m].mean(), u)
    return {"x": x, "y": y, "o": o, "eta": eta, "u": u}
