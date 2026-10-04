"""Robust Bellman operator, nominal operator, and gain-bias certificate."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rash_halpern.models import Model

DTYPE = jnp.float64


def action_scores(model: Model, v, nominal: bool = False) -> jnp.ndarray:
    """One-step scores r_ia + p^T v.

    Robust mode uses the analytic worst-case row. Nominal mode uses the
    reference row. Invalid actions are -inf.
    """
    v = jnp.asarray(v, dtype=DTYPE)
    scores = jnp.full((model.n_states, model.n_actions), -jnp.inf, dtype=DTYPE)
    for i in range(model.n_states):
        for a in range(model.n_actions):
            if not bool(model.mask[i, a]):
                continue
            row = model.row(i, a, v, nominal=nominal)
            score = model.rewards[i, a] + jnp.dot(row, v)
            scores = scores.at[i, a].set(score)
    return scores


def bellman(model: Model, v, nominal: bool = False) -> jnp.ndarray:
    """Apply T (nominal=False) or T0 (nominal=True)."""
    return jnp.max(action_scores(model, v, nominal=nominal), axis=1)


def greedy_policy(model: Model, v, nominal: bool = False) -> jnp.ndarray:
    """Greedy actions for the chosen operator.

    Exact ties use the model tie-break: smallest action index, except that
    Model III prefers risky at x. A strict improvement is never overridden.
    """
    scores = action_scores(model, v, nominal=nominal)
    return policy_from_scores(model, scores)


def policy_from_scores(model: Model, scores) -> jnp.ndarray:
    scores = jnp.asarray(scores, dtype=DTYPE)
    actions = []
    for i in range(model.n_states):
        row = scores[i]
        finite = jnp.isfinite(row)
        masked = jnp.where(finite, row, -jnp.inf)
        best = jnp.max(masked)
        # Tight absolute tolerance. Genuine gaps in these models are larger.
        tied = finite & (jnp.abs(row - best) <= 1e-10)
        prefer = int(model.prefer[i])
        if bool(tied[prefer]):
            actions.append(prefer)
        else:
            actions.append(int(jnp.argmax(tied.astype(DTYPE))))
    return jnp.asarray(actions, dtype=jnp.int32)


def gain_operator(model: Model, g) -> jnp.ndarray:
    """Reward-free continuation-gain operator (T-bar g)_i = max_a min_p p^T g."""
    out = []
    for i in range(model.n_states):
        values = [
            model.continuation_gain(i, a, g)
            for a in range(model.n_actions)
            if bool(model.mask[i, a])
        ]
        out.append(max(values))
    return jnp.asarray(out, dtype=DTYPE)


def bias_operator(model: Model, g, h) -> jnp.ndarray:
    """Gain-restricted bias operator L_g h.

    Only gain-active actions are eligible. On each of them, nature minimizes
    the bias on the gain face.
    """
    g = jnp.asarray(g, dtype=DTYPE)
    out = []
    for i in range(model.n_states):
        active = []
        for a in range(model.n_actions):
            if not bool(model.mask[i, a]):
                continue
            gain = model.continuation_gain(i, a, g)
            if gain >= float(g[i]) - 1e-9:
                active.append(model.face_bias_score(i, a, g, h))
        if not active:
            raise RuntimeError(f"no gain-active action at state {i}")
        out.append(max(active))
    return jnp.asarray(out, dtype=DTYPE)


def certificate_residual(model: Model) -> jnp.ndarray:
    """Componentwise residual of the gain-bias certificate (equation 9).

    The unrestricted identity T h = h + g* is not the certificate. Inactive
    actions can have a large transient reward, and rows off the gain face can
    look better for h alone. The paper's system is g = T-bar g and
    g + h = L_g h. The returned vector is L_g h - h - g*, which is the bias
    half. Callers that also want the gain half should use gain_operator.
    """
    g = model.g_star
    h = model.h
    lg = bias_operator(model, g, h)
    return lg - h - g
