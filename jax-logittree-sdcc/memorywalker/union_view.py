"""The literal physical-union 4D mask of Eq. 7, and where it stops being exact.

Section 4.2 defines a mask over ``H_T`` from each token's logical lifetime
``[tau(j), omega(j))``: row ``i`` admits key ``j`` iff ``tau(j) <= i < omega(j)``,
combined with a causal mask, with position ids reassigned so each row's visible
keys form a gap-free sequence.

Two departures from the equation as printed, both forced:

1. Off by one at a junction. Eq. 7's row condition is ``i < omega(j)``, but the
   row that predicts target ``t`` must condition on ``p_t(y_t)``, which drops
   every span whose junction is below ``t``, not below ``i = t - 1``. With
   ``E_1 = {y_5}`` at ``J_1 = 7``, row 7 is where ``y_8`` is read out, and
   ``y_8`` must not see ``y_5`` while row 7's own live view still does. We build
   the mask target-side: the row that reads out ``t`` admits exactly
   ``live_view(t)``. Rows that read out nothing keep their own lifetime mask.

2. Representability. A row carries one mask, so if two targets share a predictor
   row and disagree about what was live, no union mask exists. Sharing a row is
   fine as long as the two views agree, since a row can be read out twice.
   ``union_is_representable`` reports the bad case. The branch-replicated packing
   in ``exact.py`` has no such failure mode, since it replicates the row.

Even with both fixed, the union view reproduces the tree only for a one-layer
model. From layer two on, a key's hidden state has been computed under its own
row's visibility rather than under the querying branch's, so a trunk token that
should look different on two branches has one hidden state for both. This is the
gap Appendix G closes by replicating the trunk. ``test_union_view.py`` measures
it: exact at depth 1, off at depth 2.
"""

from __future__ import annotations

import jax.numpy as jnp

from .model import NEG, Config, cross_entropy, forward
from .tree import Rollout


def logical_lifetime(rollout: Rollout) -> list[tuple[int, float]]:
    """``[tau(j), omega(j))`` per physical position, ``omega = +inf`` if never evicted."""
    out = []
    for j in range(rollout.length):
        omega = float("inf")
        for e in rollout.evictions:
            if j in e.span:
                omega = min(omega, e.junction + 1)
        out.append((j, omega))
    return out


def _readout_rows(rollout: Rollout) -> dict[int, tuple[int, ...]]:
    """Predictor row per target: the last position live in that target's view."""
    return {t: max(rollout.live_view(t)) for t in rollout.response_positions()}


def union_is_representable(rollout: Rollout) -> bool:
    rows: dict[int, tuple[int, ...]] = {}
    for t, r in _readout_rows(rollout).items():
        view = rollout.live_view(t)
        if rows.setdefault(r, view) != view:
            return False
    return True


def union_mask(rollout: Rollout):
    """The ``(T, T)`` additive mask, the per-row offsets, and the loss readout."""
    if not union_is_representable(rollout):
        raise ValueError("this eviction schedule has no single-row union mask; "
                         "use the branch-replicated packing")
    n = rollout.length
    lifetimes = logical_lifetime(rollout)
    readout = _readout_rows(rollout)
    targets_of_row: dict[int, list[int]] = {}
    for t, r in readout.items():
        targets_of_row.setdefault(r, []).append(t)

    bias = [[NEG] * n for _ in range(n)]
    delta = [[0] * n for _ in range(n)]
    for i in range(n):
        if i in targets_of_row:
            visible = rollout.live_view(targets_of_row[i][0])
        else:
            visible = tuple(j for j in range(i + 1) if lifetimes[j][1] > i)
        rank = {j: p for p, j in enumerate(visible)}
        own = rank.get(i, len(visible) - 1)
        for j in visible:
            bias[i][j] = 0.0
            delta[i][j] = rank[j] - own
        if i not in rank:  # a row always attends to itself
            bias[i][i] = 0.0
            delta[i][i] = 0

    rows = [r for r in sorted(targets_of_row) for _ in targets_of_row[r]]
    targets = [rollout.tokens[t] for r in sorted(targets_of_row) for t in targets_of_row[r]]
    return (jnp.asarray(rollout.tokens), jnp.asarray(bias), jnp.asarray(delta),
            jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(targets, dtype=jnp.int32))


def union_loss(params, rollout: Rollout, cfg: Config) -> jnp.ndarray:
    tokens, bias, delta, rows, targets = union_mask(rollout)
    logits = forward(params, tokens, bias, delta, cfg)
    return cross_entropy(logits[rows], targets)
