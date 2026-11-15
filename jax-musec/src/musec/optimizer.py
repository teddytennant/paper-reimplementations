"""Musec / Soft Musec optimizer step (Algorithms 2 and 4).

Momentum is the paper's EMA
    Mc_0 = G_0
    Mc_n = (1 - beta) M_{n-1} + beta G_n   (n >= 1)
with the *clipped* buffer M_{n-1} fed back (Section 4.1). beta is the
coefficient on the new gradient, not the PyTorch-style momentum coeff.

Algorithm 2 is stated for a single matrix W. Vector / scalar leaves are
updated with the same EMA and no spectral map. The paper's LLM experiments
use AdamW for embeddings and 1D parameters (Section 6.1); that fallback is
not implemented here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from musec.clip import hard_spectral_clip, soft_spectral_clip

PyTree = Any


class MusecState(NamedTuple):
    momentum: PyTree
    step: Array  # int32 scalar; 0 means next update is n = 0


def init(params: PyTree) -> MusecState:
    momentum = jax.tree.map(jnp.zeros_like, params)
    return MusecState(momentum=momentum, step=jnp.array(0, dtype=jnp.int32))


def _clip_leaf(M: Array, D: float, mode: str, ns_steps: int) -> Array:
    if M.ndim != 2:
        return M
    if mode == "hard":
        return hard_spectral_clip(M, D)
    if mode == "soft":
        return soft_spectral_clip(M, D, ns_steps=ns_steps)
    raise ValueError(f"unknown mode {mode!r}; expected 'hard' or 'soft'")


def step(
    params: PyTree,
    grads: PyTree,
    state: MusecState,
    *,
    eta: float,
    beta: float,
    D: float,
    mode: str = "hard",
    ns_steps: int = 5,
    weight_decay: float = 0.0,
) -> tuple[PyTree, MusecState]:
    """One Musec (mode='hard') or Soft Musec (mode='soft') update.

    Parameters
    ----------
    eta : learning rate
    beta : EMA coefficient on the new gradient (Algorithm 2)
    D : clipping threshold
    ns_steps : Newton–Schulz iterations K in Algorithm 3 (paper uses 5)
    weight_decay : optional decoupled decay applied as W <- (1 - eta * wd) W
        before the Musec step. Not part of Algorithm 2; used in Section 6.
    """

    def mix(m_prev: Array, g: Array) -> Array:
        return jnp.where(state.step == 0, g, (1.0 - beta) * m_prev + beta * g)

    mixed = jax.tree.map(mix, state.momentum, grads)
    clipped = jax.tree.map(lambda m: _clip_leaf(m, D, mode, ns_steps), mixed)

    def apply(w: Array, m: Array) -> Array:
        w_decayed = w * (1.0 - eta * weight_decay)
        return w_decayed - eta * m

    new_params = jax.tree.map(apply, params, clipped)
    new_state = MusecState(momentum=clipped, step=state.step + 1)
    return new_params, new_state
