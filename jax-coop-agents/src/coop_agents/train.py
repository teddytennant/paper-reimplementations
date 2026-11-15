"""One parallel SGD/Adam step per agent with stop-gradient consensus."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax

from coop_agents.consensus import (
    majority,
    majority_loo,
    simplicial,
    voter,
    weighted_average,
)
from coop_agents.losses import (
    combined_loss,
    confidence,
    group_loss,
    isolated_loss,
)
from coop_agents.models import apply_mlp, softmax

# Table I names plus the task's "weighted average" alias of simplicial consensus.
STRATEGIES = (
    "isolated",
    "voter",
    "majority",
    "majority_loo",
    "weighted_average",
    "simplicial",
    "simplicial_loo",
)

_CONFIDENCE_SCALE = {
    "voter",
    "majority",
    "majority_loo",
    "simplicial",
    "simplicial_loo",
}


def init_opt_state(
    agents: Sequence,
    lr: float = 0.1,
    kind: str = "sgd",
    key: jax.Array | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-7,
) -> dict:
    """SGD or Adam state. Paper Table IV uses Adam; smoke/tests default to SGD."""
    if key is None:
        key = jax.random.PRNGKey(0)
    state = {
        "kind": kind,
        "lr": jnp.asarray(lr),
        "key": key,
        "t": jnp.array(0, dtype=jnp.int32),
        "beta1": jnp.asarray(beta1),
        "beta2": jnp.asarray(beta2),
        "eps": jnp.asarray(eps),
    }
    if kind == "adam":
        state["m"] = jax.tree.map(jnp.zeros_like, list(agents))
        state["v"] = jax.tree.map(jnp.zeros_like, list(agents))
    return state


def _agent_probs(agents: Sequence, x: jax.Array) -> jax.Array:
    logits = jnp.stack([apply_mlp(p, x) for p in agents], axis=0)
    return softmax(logits)


def _consensus(
    strategy: str,
    probs_sg: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array | None]:
    """Return (per-agent or shared consensus, optional global consensus)."""
    w = confidence(probs_sg)
    glob = None
    if strategy == "isolated":
        cons = jnp.zeros_like(probs_sg[0])
    elif strategy == "voter":
        cons = voter(probs_sg, key)
    elif strategy == "majority":
        cons = majority(probs_sg)
    elif strategy == "majority_loo":
        cons = majority_loo(probs_sg)
    elif strategy == "weighted_average":
        cons = weighted_average(probs_sg, w)
    elif strategy == "simplicial":
        cons = simplicial(probs_sg, w, loo=False)
        glob = cons
    elif strategy == "simplicial_loo":
        cons = simplicial(probs_sg, w, loo=True)
        glob = simplicial(probs_sg, w, loo=False)
    else:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    return cons, glob


def _mean_ce(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(isolated_loss(pred, target))


def agent_combined_loss(
    params,
    x: jax.Array,
    y: jax.Array,
    cons_k: jax.Array,
    w_k: jax.Array,
    strategy: str,
    lam: float | jax.Array,
    glob_cons: jax.Array | None = None,
) -> jax.Array:
    """L_k = L_isol * (L_isol + λ L^G) for one agent; consensus is a snapshot."""
    logits = apply_mlp(params, x)
    probs = softmax(logits)
    isol_n = isolated_loss(probs, y)
    if strategy == "isolated":
        group_n = jnp.zeros_like(isol_n)
    else:
        group_n = group_loss(probs, cons_k)
        if strategy in _CONFIDENCE_SCALE:
            group_n = (1.0 - w_k) * group_n
        if glob_cons is not None:
            group_n = group_n + isolated_loss(glob_cons, y)
    return jnp.mean(combined_loss(isol_n, group_n, lam))


def step(agents, batch, strategy, lam, opt_state):
    """One parallel update of every agent on the current consensus snapshot.

    Consensus is stop-gradient: group terms do not backprop into other agents.
    """
    x, y = batch
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    agents = list(agents)
    strategy = str(strategy)
    key = opt_state["key"]
    key, sub = jax.random.split(key)

    probs = _agent_probs(agents, x)
    probs_sg = lax.stop_gradient(probs)
    w_sg = lax.stop_gradient(confidence(probs_sg))
    cons, glob = _consensus(strategy, probs_sg, sub)
    cons = lax.stop_gradient(cons)
    glob = None if glob is None else lax.stop_gradient(glob)

    new_agents = []
    grads = []
    combined_vals = []
    isol_vals = []
    for k, params in enumerate(agents):
        if strategy in ("majority_loo", "simplicial_loo"):
            cons_k = cons[k]
        else:
            cons_k = cons
        w_k = w_sg[k]

        def loss_fn(p, cons_k=cons_k, w_k=w_k):
            return agent_combined_loss(p, x, y, cons_k, w_k, strategy, lam, glob)

        val, g = jax.value_and_grad(loss_fn)(params)
        grads.append(g)
        new_agents.append(params)
        combined_vals.append(val)
        isol_vals.append(_mean_ce(probs[k], y))

    lr = opt_state["lr"]
    kind = opt_state["kind"]
    t = opt_state["t"] + 1
    new_state = dict(opt_state)
    new_state["key"] = key
    new_state["t"] = t

    if kind == "adam":
        beta1, beta2, eps = opt_state["beta1"], opt_state["beta2"], opt_state["eps"]
        m = opt_state["m"]
        v = opt_state["v"]
        new_m = jax.tree.map(lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, grads)
        new_v = jax.tree.map(lambda vi, gi: beta2 * vi + (1.0 - beta2) * (gi * gi), v, grads)
        mhat = jax.tree.map(lambda mi: mi / (1.0 - beta1**t), new_m)
        vhat = jax.tree.map(lambda vi: vi / (1.0 - beta2**t), new_v)
        new_agents = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            agents,
            mhat,
            vhat,
        )
        new_state["m"] = new_m
        new_state["v"] = new_v
    else:
        new_agents = jax.tree.map(lambda p, g: p - lr * g, agents, grads)

    pred = jnp.argmax(probs, axis=-1)
    y_idx = y if y.ndim == 1 or y.shape[-1] != probs.shape[-1] else jnp.argmax(y, axis=-1)
    acc = jnp.mean((pred == y_idx).astype(jnp.float32))
    metrics = {
        "combined": jnp.stack(combined_vals),
        "isolated": jnp.stack(isol_vals),
        "acc": acc,
        "probs": probs,
    }
    return list(new_agents), new_state, metrics
