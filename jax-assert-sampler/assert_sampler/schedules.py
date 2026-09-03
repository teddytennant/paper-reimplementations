"""Stochasticity schedules eta_i, Equations 12-15 of arXiv:2609.00955.

Index i runs 0 .. S-1 with i = 0 the early, high-noise step, and normalized
denoising progress is r_i = i / (S - 1). Dividing by S - 1 rather than S is what
forces eta_{S-1} = 0, so the last update is always deterministic.
"""

from __future__ import annotations

import jax.numpy as jnp


def progress(num_steps: int) -> jnp.ndarray:
    """r_i = i / (S - 1), Section 3.3."""
    if num_steps < 2:
        raise ValueError("the eta schedules are defined for S > 1")
    return jnp.arange(num_steps) / (num_steps - 1)


def cosine_decay(p: jnp.ndarray) -> jnp.ndarray:
    """D(p) = (1 + cos(pi p)) / 2, Equation 13."""
    return 0.5 * (1.0 + jnp.cos(jnp.pi * p))


def assert_schedule(num_steps: int, eta_max: float = 0.8,
                    tau_trans: float = 0.1) -> jnp.ndarray:
    """ASSERT's eta_i, Equation 12.

    eta_i = eta_max for r_i < tau_trans, and
    eta_i = eta_max * D((r_i - tau_trans) / (1 - tau_trans)) afterwards.

    The defaults are the paper's: eta_max = 0.8 with a 10% early plateau, picked
    from the CIFAR-10 ablation in Table 2 and then held fixed across datasets.
    """
    if not 0.0 <= tau_trans < 1.0:
        raise ValueError("tau_trans must lie in [0, 1)")
    r = progress(num_steps)
    decayed = eta_max * cosine_decay((r - tau_trans) / (1.0 - tau_trans))
    return jnp.where(r < tau_trans, eta_max, decayed)


def constant_schedule(num_steps: int, eta: float) -> jnp.ndarray:
    """eta_i = eta everywhere. eta = 0 is deterministic DDIM; eta = 1 is the
    DDPM-like limit of the DDIM family."""
    return jnp.full((num_steps,), eta)


def linear_decay_schedule(num_steps: int, eta_max: float = 0.8,
                          eta_min: float = 0.0) -> jnp.ndarray:
    """Equation 14, the linear-decay baseline."""
    r = progress(num_steps)
    return eta_max - r * (eta_max - eta_min)


def step_schedule(num_steps: int, eta_max: float = 0.8, eta_min: float = 0.0,
                  tau_trans: float = 0.4) -> jnp.ndarray:
    """Step decay, the left half of Equation 15. The paper evaluates it at
    tau_trans = 0.4."""
    r = progress(num_steps)
    return jnp.where(r < tau_trans, eta_max, eta_min)


def reverse_step_schedule(num_steps: int, eta_max: float = 0.8,
                          eta_min: float = 0.0,
                          tau_trans: float = 0.4) -> jnp.ndarray:
    """Reverse step, the right half of Equation 15. It saves the stochasticity for
    final refinement, and is the control that isolates *when* stochasticity
    helps."""
    r = progress(num_steps)
    return jnp.where(r < tau_trans, eta_min, eta_max)


def window_schedule(num_steps: int, eta: float, start: int, stop: int) -> jnp.ndarray:
    """Constant eta on step indices [start, stop) and 0 outside.

    This is the Figure 5 diagnostic, not ASSERT.
    """
    i = jnp.arange(num_steps)
    return jnp.where((i >= start) & (i < stop), eta, 0.0)
