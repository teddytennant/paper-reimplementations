"""Overdamped Langevin, two-way shooting, HAB, and path weights."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from genaimmd.potential import BETA, D, DT, KBIAS, force, state_label


def euler_maruyama(x, key, dt: float = DT, D: float = D, force_fn=None):
    """One Euler-Maruyama step of overdamped Langevin (kBT=1)."""
    if force_fn is None:
        force_fn = force
    F = force_fn(x)
    noise = jax.random.normal(key, x.shape, dtype=x.dtype)
    return x + D * F * dt + jnp.sqrt(2.0 * D * dt) * noise


def integrate(x, key, n_steps: int, dt: float = DT, D: float = D, force_fn=None):
    keys = jax.random.split(key, n_steps)

    def body(x, k):
        x = euler_maruyama(x, k, dt=dt, D=D, force_fn=force_fn)
        return x, x

    x, xs = jax.lax.scan(body, x, keys)
    return x, xs


def shoot_until_state(
    x,
    key,
    max_steps: int = 500,
    dt: float = DT,
    D: float = D,
    force_fn=None,
):
    """Integrate until A or B is hit (or max_steps). Returns end, label, length."""
    x = jnp.asarray(x)
    lab0 = state_label(x)
    hit0 = lab0 != 0

    def body(t, carry):
        x, key, lab, length, hit = carry
        key, k = jax.random.split(key)
        x_new = euler_maruyama(x, k, dt=dt, D=D, force_fn=force_fn)
        x = jnp.where(hit, x, x_new)
        lab_new = state_label(x)
        newly = (~hit) & (lab_new != 0)
        lab = jnp.where(newly, lab_new, lab)
        length = jnp.where(newly, t + 1, length)
        hit = hit | newly
        return x, key, lab, length, hit

    length0 = jnp.array(0, dtype=jnp.int32)
    x, key, lab, length, hit = jax.lax.fori_loop(
        0, max_steps, body, (x, key, lab0, length0, hit0)
    )
    return x, lab, length


def two_way_shoot(
    x,
    key,
    max_steps: int = 500,
    dt: float = DT,
    D: float = D,
    force_fn=None,
):
    """Two independent Brownian trajectories (overdamped: no momentum reversal).

    HAB=1 iff the two ends hit different states. Labels s=+1 if A (or neither),
    s=-1 if B first, matching AIMMD eq. (6).
    """
    k1, k2 = jax.random.split(key)
    x1, lab1, n1 = shoot_until_state(x, k1, max_steps, dt, D, force_fn)
    x2, lab2, n2 = shoot_until_state(x, k2, max_steps, dt, D, force_fn)
    different = (lab1 != 0) & (lab2 != 0) & (lab1 != lab2)
    hab = different.astype(jnp.float32)
    # s_ij = -1 if B first else +1
    s1 = jnp.where(lab1 == -1, -1.0, 1.0)
    s2 = jnp.where(lab2 == -1, -1.0, 1.0)
    return {
        "ends": jnp.stack([x1, x2], axis=0),
        "labels": jnp.stack([s1, s2]),
        "state_hits": jnp.stack([lab1, lab2]),
        "lengths": jnp.stack([n1, n2]),
        "hab": hab,
        "L": (n1 + n2).astype(jnp.float32),
    }


def shooting_acceptance(L, L_prime, hab_prime):
    """Uniform shooting P_acc = min(1, L/L') if HAB(X')=1 else 0, eq. (2)."""
    L = jnp.asarray(L, dtype=jnp.float32)
    L_prime = jnp.asarray(L_prime, dtype=jnp.float32)
    hab_prime = jnp.asarray(hab_prime, dtype=jnp.float32)
    acc = jnp.minimum(1.0, L / jnp.maximum(L_prime, 1e-12))
    return jnp.where(hab_prime > 0, acc, jnp.zeros_like(acc))


def path_weight(pb_along_path, pB_hat: float = 0.5, kbias: float = KBIAS, beta: float = BETA):
    """Ω(X) from eq. (12) for generated shooting points at p̂B (default 0.5)."""
    pb = jnp.asarray(pb_along_path)
    terms = jnp.exp(-beta * (kbias / 2.0) * jnp.square(pb - pB_hat))
    return 1.0 / jnp.sum(terms)
