"""Frequency-to-pixel schedule for the joint objective, Eq. (4).

The paper sets w_f = lambda(s) and w_v = 1 - lambda(s), where lambda(s) is a
sigmoid decaying from 1 to 0 centered on the crossover point s*. s* is picked by
a crossing-point rule and not by grid search: the step where the f-loss and
v-loss validation FID curves intersect, about 200k steps for the B and L models
and about 350k for XL (Appendix A). The paper never states the sigmoid's
temperature, so ``width`` here is a free parameter; it is the number of steps
over which lambda falls from about 0.88 to about 0.12.

Table 4 ablates the alternatives. Constant (w_f, w_v) pairs all lose to the
sigmoid, and the reversed sigmoid, pixels first then frequencies, is among the
worst settings tested, which is the direction claim this schedule encodes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def frequency_weight(step, switch_step, width):
    """lambda(s), decaying from 1 to 0 through 0.5 at ``switch_step``."""
    return jax.nn.sigmoid(-(jnp.asarray(step) - switch_step) / width)


def loss_weights(step, switch_step, width):
    """(w_f, w_v) = (lambda(s), 1 - lambda(s)) for Eq. (4)."""
    lam = frequency_weight(step, switch_step, width)
    return lam, 1.0 - lam
