"""Fisher-diagonal trust-region sides and the TuRBO-style length schedule."""

import math

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp


KAPPA_INIT = 0.8
KAPPA_MAX = 1.6
KAPPA_MIN = 0.5 ** 7
SUCCESS_LIMIT = 10
Q = 1


def sides(g_diag, kappa):
    """Volume-normalized side lengths from a diagonal pullback Fisher.

    eps = mean(g) + 1e-12, w_i = 1 / (g_i + eps),
    side_i = kappa * w_i / (prod_j w_j)^(1/d), so prod(side) = kappa^d.
    A small g_i (low sensitivity) receives a longer side.
    """
    g = jnp.asarray(g_diag)
    eps = jnp.mean(g) + 1e-12
    weights = 1.0 / (g + eps)
    log_geom = jnp.mean(jnp.log(weights))
    return kappa * jnp.exp(jnp.log(weights) - log_geom)


def box(center, sides):
    """Axis-aligned box [center - side/2, center + side/2] clipped to [0, 1]."""
    center = jnp.asarray(center)
    half = 0.5 * jnp.asarray(sides)
    lo = jnp.clip(center - half, 0.0, 1.0)
    hi = jnp.clip(center + half, 0.0, 1.0)
    return lo, hi


def _fail_limit(d):
    return min(math.ceil(max(4.0 / Q, float(d) / Q)), 20)


def step_length(kappa, improved, succ, fail, d):
    """Update kappa from one strict-improvement bit.

    After 10 consecutive strict improvements, kappa <- min(kappa_max, 2 kappa)
    and the success count resets. After fail_limit consecutive
    non-improvements, kappa <- kappa / 2 and the failure count resets.
    If an update drops kappa below kappa_min = 0.5**7, restart at kappa_init
    and zero both counts. The other counter resets on every opposite outcome.
    """
    kappa = float(kappa)
    succ = int(succ)
    fail = int(fail)
    improved = bool(improved)
    restarted = False
    if improved:
        succ += 1
        fail = 0
        if succ >= SUCCESS_LIMIT:
            kappa = min(KAPPA_MAX, 2.0 * kappa)
            succ = 0
            if kappa < KAPPA_MIN:
                kappa = KAPPA_INIT
                succ = 0
                fail = 0
                restarted = True
    else:
        fail += 1
        succ = 0
        if fail >= _fail_limit(d):
            kappa = kappa / 2.0
            fail = 0
            if kappa < KAPPA_MIN:
                kappa = KAPPA_INIT
                succ = 0
                fail = 0
                restarted = True
    return kappa, succ, fail, restarted
