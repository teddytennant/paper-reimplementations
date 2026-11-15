"""MSDG-Prune visual token selection (Algorithm 1, arXiv:2609.36916).

The caller passes encoder states, projected visual tokens, and query embeddings.
This module does not run an MLLM, load images, apply the Qwen spatial merger,
or run the Mini-Gemini dual encoder.

States are indexed 0..L as in the algorithm. Returned token indices are 0-based.
Assignment of a token to tied centroids uses the lower centroid index. The paper
does not specify that tie. Empty groups are filled in ascending centroid index
with distinct token directions in ascending margin order (ties: original token
order). The paper specifies the direction order but not the empty-group index map.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Appendix C.1 presets. The functions take d_s and tau explicitly.
CLIP_VIT_L_D_S = 650
QWEN25_VL_D_S = 849
DEFAULT_TAU = 50.0

# Appendix C.2. Windows are (ell_s, ell_e).
CLIP_SALIENCY_WINDOW = (14, 19)
QWEN_SALIENCY_WINDOW = (19, 24)

DEFAULT_K = 20
DEFAULT_SEED = 0
DEFAULT_ELL_0 = 2
MAX_KMEANS_ITERS = 10
LOSS_TOL = 1e-5
UNIT_EPS = 1e-12


class MSDGPruneResult(NamedTuple):
    """Retained tokens and the grouping the allocator used.

    indices: retained visual-token indices, sorted in original order.
    tokens: corresponding rows of the projected visual tokens.
    group_ids: length N, 0-based group for candidates, -1 if excluded by the sink filter.
    budgets: length K integer budgets, one per group.
    """

    indices: jnp.ndarray
    tokens: jnp.ndarray
    group_ids: jnp.ndarray
    budgets: jnp.ndarray


def unit(z) -> jnp.ndarray:
    """unit(z) = z / max(||z||, 1e-12). z is one vector."""
    z = jnp.asarray(z, dtype=jnp.float64)
    if z.ndim != 1:
        raise ValueError("unit expects a single vector")
    norm = jnp.linalg.norm(z)
    return z / jnp.maximum(norm, UNIT_EPS)


def _unit_rows(z) -> jnp.ndarray:
    """Same rule as unit, applied to each row. A 1-d vector is one row."""
    z = jnp.asarray(z, dtype=jnp.float64)
    norms = jnp.linalg.norm(z, axis=-1, keepdims=True)
    return z / jnp.maximum(norms, UNIT_EPS)


def update_direction(h_late, h_early) -> jnp.ndarray:
    """Equation (1): unit(unit(h_late) - unit(h_early))."""
    return unit(unit(h_late) - unit(h_early))


def endpoint_displacement(h_end, h_start) -> jnp.ndarray:
    """L2 gap between saliency-window endpoints, not a sum of per-layer norms."""
    diff = jnp.asarray(h_end, dtype=jnp.float64) - jnp.asarray(h_start, dtype=jnp.float64)
    return jnp.linalg.norm(diff)


def max_query_cosine(visual_token, query_embeddings) -> jnp.ndarray:
    """Max cosine between one projected token and the query embeddings."""
    queries = jnp.asarray(query_embeddings, dtype=jnp.float64)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2 or queries.shape[0] == 0:
        raise ValueError("query set must be nonempty")
    return jnp.max(_unit_rows(visual_token) @ _unit_rows(queries).T)


def _softmax(logits) -> jnp.ndarray:
    """Softmax. Max-subtracted softmax is the same function."""
    logits = jnp.asarray(logits, dtype=jnp.float64)
    shifted = logits - jnp.max(logits)
    weights = jnp.exp(shifted)
    return weights / jnp.sum(weights)


def farthest_point_init(directions, n_groups, seed=DEFAULT_SEED):
    """Deterministic farthest-point sampling, Equation (5).

    Rows of directions are candidates v_0 .. v_{M-1} in original order.
    The first centroid is row (seed mod M). Each next centroid maximizes
    1 - max_{r < c} d_i dot mu_r over unchosen rows. Ties keep the earlier row.
    Returns (chosen row indices, centroids).
    """
    directions = jnp.asarray(directions, dtype=jnp.float64)
    if directions.ndim != 2 or directions.shape[0] == 0:
        raise ValueError("directions must have shape (M, D) with M >= 1")
    m = int(directions.shape[0])
    k = int(n_groups)
    if k < 1 or k > m:
        raise ValueError(f"need 1 <= K <= |V|, got K={k}, |V|={m}")
    chosen = []
    centroids = []
    taken = jnp.zeros((m,), dtype=bool)
    first = int(seed) % m
    chosen.append(first)
    centroids.append(directions[first])
    taken = taken.at[first].set(True)
    for _ in range(1, k):
        cents = jnp.stack(centroids, axis=0)
        max_dot = jnp.max(directions @ cents.T, axis=1)
        score = 1.0 - max_dot
        score = jnp.where(taken, -jnp.inf, score)
        nxt = int(jnp.argmax(score))
        chosen.append(nxt)
        centroids.append(directions[nxt])
        taken = taken.at[nxt].set(True)
    return jnp.array(chosen, dtype=jnp.int32), jnp.stack(centroids, axis=0)


def _assign(directions, centroids):
    """Assign each row to the centroid with maximum dot. Ties: lower index.

    jnp.argmax returns the first maximum, which is the lower centroid index.
    The paper does not specify this tie.
    """
    sims = directions @ centroids.T
    labels = jnp.argmax(sims, axis=1)
    margins = jnp.max(sims, axis=1)
    loss = float(jnp.mean(1.0 - margins))
    return labels, margins, loss


def _update_centroids(directions, centroids, labels, margins):
    """Nonempty groups become unit(sum d_i). Empty groups take distinct d_i.

    Empty groups are visited in ascending centroid index. They receive distinct
    token directions in ascending m_i order, ties broken by original token order.
    """
    k = int(centroids.shape[0])
    updated = centroids
    empty = []
    for c in range(k):
        members = directions[labels == c]
        if int(members.shape[0]) == 0:
            empty.append(c)
            continue
        updated = updated.at[c].set(unit(jnp.sum(members, axis=0)))
    if empty:
        order = sorted(range(int(directions.shape[0])), key=lambda i: (float(margins[i]), i))
        for slot, c in enumerate(empty):
            updated = updated.at[c].set(directions[order[slot]])
    return updated


def spherical_kmeans(directions, n_groups, seed=DEFAULT_SEED, max_iterations=MAX_KMEANS_ITERS):
    """Spherical k-means from Algorithm 1, at most 10 iterations.

    Stop when t > 1 and (labels are unchanged or the absolute relative loss
    change is below 1e-5), or after max_iterations. Centroids are updated only
    when the loop does not break. Tokens are reassigned to the final centroids.
    Returns (labels, centroids).
    """
    directions = jnp.asarray(directions, dtype=jnp.float64)
    _, centroids = farthest_point_init(directions, n_groups, seed)
    labels = None
    prev_labels = None
    loss_prev = None
    for t in range(1, int(max_iterations) + 1):
        labels, margins, loss = _assign(directions, centroids)
        if t > 1:
            unchanged = bool(jnp.array_equal(labels, prev_labels))
            rel = abs(loss_prev - loss) / max(loss_prev, UNIT_EPS)
            if unchanged or rel < LOSS_TOL:
                break
        centroids = _update_centroids(directions, centroids, labels, margins)
        prev_labels = labels
        loss_prev = loss
    labels, _, _ = _assign(directions, centroids)
    return labels, centroids


def allocate_budgets(proportions, sizes, budget) -> jnp.ndarray:
    """Integer budgets from Algorithm 1 lines 16-19.

    b_c = min(|C_c|, floor(B * p_c)), then one pass (not a cycle) over groups
    with room, in descending fractional part, ties lower c, adding one while
    R > 0. If R remains, visit groups in descending p_c, ties lower c, and add
    min(R, remaining capacity).
    """
    proportions = [float(p) for p in proportions]
    sizes = [int(s) for s in sizes]
    if len(proportions) != len(sizes):
        raise ValueError("proportions and sizes must have the same length")
    budget_f = float(budget)
    if budget_f != math.floor(budget_f):
        raise ValueError("budget must be an integer")
    budget = int(budget_f)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    raw = [budget * p for p in proportions]
    floors = [math.floor(v) for v in raw]
    b = [min(sizes[c], floors[c]) for c in range(len(sizes))]
    remainder = budget - sum(b)
    frac = [raw[c] - floors[c] for c in range(len(sizes))]
    # One pass. Do not cycle back to the first group.
    eligible = [c for c in range(len(sizes)) if b[c] < sizes[c]]
    eligible.sort(key=lambda c: (-frac[c], c))
    for c in eligible:
        if remainder <= 0:
            break
        if b[c] < sizes[c]:
            b[c] += 1
            remainder -= 1
    if remainder > 0:
        order = sorted(range(len(sizes)), key=lambda c: (-proportions[c], c))
        for c in order:
            if remainder <= 0:
                break
            delta = min(remainder, sizes[c] - b[c])
            if delta > 0:
                b[c] += delta
                remainder -= delta
    return jnp.array(b, dtype=jnp.int32)


def _layer_count(states) -> int:
    return int(states.shape[1])


def _require_layer(name, ell, n_layers) -> int:
    ell = int(ell)
    if ell < 0 or ell >= n_layers:
        raise ValueError(f"{name}={ell} is outside state layers 0..{n_layers - 1}")
    return ell


def _select_within_groups(candidate_ids, labels, scores, budgets):
    """Top-b scores in each group. Ties keep the earlier original index."""
    selected = []
    k = int(budgets.shape[0])
    for c in range(k):
        members = [i for i in range(int(labels.shape[0])) if int(labels[i]) == c]
        members.sort(key=lambda i: (-float(scores[i]), int(candidate_ids[i])))
        take = int(budgets[c])
        for i in members[:take]:
            selected.append(int(candidate_ids[i]))
    selected.sort()
    return selected


def msdg_prune(
    states,
    visual_tokens,
    query_embeddings,
    budget,
    ell_star,
    n_groups=DEFAULT_K,
    ell_s=CLIP_SALIENCY_WINDOW[0],
    ell_e=CLIP_SALIENCY_WINDOW[1],
    ell_0=DEFAULT_ELL_0,
    ell_d=None,
    d_s=CLIP_VIT_L_D_S,
    tau=DEFAULT_TAU,
    seed=DEFAULT_SEED,
) -> MSDGPruneResult:
    """Select B visual tokens with Algorithm 1.

    states: (N, L+1, D) encoder states indexed 0..L.
    visual_tokens: (N, D_x) projected tokens, aligned with states.
    query_embeddings: (Q, D_x), or one vector of shape (D_x,).
    Default grouping endpoints are ell_0 = 2 and ell_d = L - 1.
    Default saliency window is the CLIP preset 14 to 19. Pass
    QWEN_SALIENCY_WINDOW for the Qwen preset 19 to 24.
    d_s presets are CLIP_VIT_L_D_S = 650 and QWEN25_VL_D_S = 849.
    A feature dimension smaller than d_s is a caller error.
    """
    states = jnp.asarray(states)
    visual_tokens = jnp.asarray(visual_tokens)
    queries = jnp.asarray(query_embeddings)
    if states.ndim != 3:
        raise ValueError("states must have shape (N, L+1, D)")
    if visual_tokens.ndim != 2:
        raise ValueError("visual_tokens must have shape (N, D_x)")
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2 or int(queries.shape[0]) == 0:
        raise ValueError("query set must be nonempty")
    n = int(states.shape[0])
    if int(visual_tokens.shape[0]) != n:
        raise ValueError("visual_tokens and states must have the same token count")
    if int(queries.shape[1]) != int(visual_tokens.shape[1]):
        raise ValueError("query embeddings and visual tokens must share a feature size")

    n_layers = _layer_count(states)
    feature_dim = int(states.shape[2])
    d_s = int(d_s)
    if d_s < 0 or d_s >= feature_dim:
        raise ValueError(
            f"d_s={d_s} is outside feature dim {feature_dim}; a short feature axis is a caller error"
        )
    ell_star = _require_layer("ell_star", ell_star, n_layers)
    sink_layer = ell_star + 1
    if sink_layer >= n_layers:
        raise ValueError(f"ell_star+1={sink_layer} is outside state layers 0..{n_layers - 1}")
    ell_s = _require_layer("ell_s", ell_s, n_layers)
    ell_e = _require_layer("ell_e", ell_e, n_layers)
    ell_0 = _require_layer("ell_0", ell_0, n_layers)
    if ell_d is None:
        # Indices 0..L, so L = n_layers - 1 and the default ell_d is L - 1.
        ell_d = n_layers - 2
    ell_d = _require_layer("ell_d", ell_d, n_layers)

    keep = jnp.abs(states[:, sink_layer, d_s]) <= float(tau)
    candidate_ids = jnp.where(keep)[0]
    m = int(candidate_ids.shape[0])
    k = int(n_groups)
    budget_f = float(budget)
    if budget_f != math.floor(budget_f):
        raise ValueError("budget must be an integer")
    budget = int(budget_f)
    if budget < 0 or budget > m:
        raise ValueError(f"need 0 <= B <= |V|, got B={budget}, |V|={m}")
    if k < 1 or k > m:
        raise ValueError(f"need 1 <= K <= |V|, got K={k}, |V|={m}")

    states64 = states.astype(jnp.float64)
    late = states64[candidate_ids, ell_d, :]
    early = states64[candidate_ids, ell_0, :]
    directions = _unit_rows(_unit_rows(late) - _unit_rows(early))
    if not float(jnp.max(jnp.linalg.norm(directions, axis=-1))) > 0.0:
        raise ValueError("max_i ||d_i|| must be > 0")

    # Endpoint gap only. Do not sum norms along the path between ell_s and ell_e.
    displacement = jnp.linalg.norm(
        states64[candidate_ids, ell_e, :] - states64[candidate_ids, ell_s, :],
        axis=-1,
    )
    relevance = jnp.max(
        _unit_rows(visual_tokens[candidate_ids]) @ _unit_rows(queries).T,
        axis=-1,
    )
    scores = relevance * displacement

    labels, _centroids = spherical_kmeans(directions, k, seed)
    group_scores = []
    sizes = []
    for c in range(k):
        member_scores = scores[labels == c]
        sizes.append(int(member_scores.shape[0]))
        if sizes[-1] == 0:
            group_scores.append(0.0)
        else:
            group_scores.append(float(jnp.mean(member_scores)))
    proportions = _softmax(jnp.array(group_scores, dtype=jnp.float64))
    budgets = allocate_budgets(proportions, sizes, budget)
    if int(jnp.sum(budgets)) != budget:
        raise RuntimeError("budget allocator did not assign B tokens")

    selected = _select_within_groups(candidate_ids, labels, scores, budgets)
    indices = jnp.array(selected, dtype=jnp.int32)
    group_ids = jnp.full((n,), -1, dtype=jnp.int32)
    group_ids = group_ids.at[candidate_ids].set(labels.astype(jnp.int32))
    return MSDGPruneResult(
        indices=indices,
        tokens=visual_tokens[indices],
        group_ids=group_ids,
        budgets=budgets,
    )
