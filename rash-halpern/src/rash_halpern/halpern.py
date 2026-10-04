"""Algorithm 1: approximately shifted robust Halpern iteration.

The update is the printed two-phase recurrence, not a generic Halpern step.
Phase I estimates the gain by x_N / N with x_0 = 0. Phase II anchors at x_N
and shifts each Bellman update by that estimate:

    z_{t+1} = (2/(t+3)) z_0 + ((t+1)/(t+3)) (T z_t - g_hat_N)
"""

from __future__ import annotations

import os
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rash_halpern.models import Model
from rash_halpern.operator import action_scores, bellman, policy_from_scores

DTYPE = jnp.float64


class Algorithm1Result(NamedTuple):
    g_hat: jnp.ndarray
    Z: jnp.ndarray
    displacement: jnp.ndarray
    pi: jnp.ndarray


def algorithm1(model: Model, N: int, nominal: bool = False) -> Algorithm1Result:
    """Run Algorithm 1 with T, or the same update with T replaced by T0.

    Always starts at x_0 = 0. The extracted controller is greedy for the same
    operator that drove the iteration, evaluated at Z_N. Ties follow the
    model rule (favor c in Models I and II, favor risky in Model III).
    """
    if int(N) < 1:
        raise ValueError("N must be an integer >= 1")
    N = int(N)
    x = jnp.zeros(model.n_states, dtype=DTYPE)
    for _j in range(N):
        x = bellman(model, x, nominal=nominal)
    g_hat = x / jnp.asarray(N, dtype=DTYPE)
    z0 = x
    z = z0
    for t in range(N):
        tz = bellman(model, z, nominal=nominal)
        # Printed weights: 2/(t+3) on the anchor and (t+1)/(t+3) on the shift.
        z = (2.0 / (t + 3.0)) * z0 + ((t + 1.0) / (t + 3.0)) * (tz - g_hat)
    Z = z
    scores = action_scores(model, Z, nominal=nominal)
    TZ = jnp.max(scores, axis=1)
    displacement = TZ - Z
    pi = policy_from_scores(model, scores)
    return Algorithm1Result(g_hat=g_hat, Z=Z, displacement=displacement, pi=pi)
