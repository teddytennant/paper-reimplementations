"""The two exact walks of Section 4: LogitTree and the packed 4D mask.

Both score every target on the unique branch whose prefix equals the live view
it was decoded from, so both have zero conditioning bias, and by Theorem 1 they
are the same walk at different levels of the stack.

Which packing is "the" 4D mask. Section 4.2 defines the mask over the physical
union H_T, one row per physical token (Eq. 7). Appendix G says the mask the
training loop actually runs is the branch-replicated one: one copy of each
shared-trunk token per branch, block-diagonal across branches, position ids
restarting per branch, and calls the union form "the row-wise characterization
of what the block-packed execution computes". Those two are not the same
function once the model has more than one layer, because in the union view a
key's hidden state is computed under its own row's visibility rather than under
the querying branch's. We take Appendix G's branch-replicated packing as the
4D-mask path, since that is the reading under which Theorem 1's gradient
equivalence is true at any depth. The literal union view is in
``union_view.py`` together with a test of exactly where it stops matching.
"""

from __future__ import annotations

import jax.numpy as jnp

from .model import NEG, Config, cross_entropy, forward, sequence_logits
from .tree import Rollout


def branch_logits(params, rollout: Rollout, k: int, cfg: Config) -> jnp.ndarray:
    """One forward over branch ``k``, ``Logits_pi = f_theta(H'^pi)`` of Eq. 6."""
    ids = jnp.asarray([rollout.tokens[i] for i in rollout.branch(k)])
    return sequence_logits(params, ids, cfg)


def branch_readout(rollout: Rollout, k: int) -> tuple[list[int], list[int], list[int]]:
    """Rows of branch ``k`` that carry loss, their targets, and the physical positions.

    Branch k restricted to positions below t is exactly ``live_view(t)`` for every
    target t the layer owns, so the row just before t on the branch predicts t
    under the conditioning it was generated with. Every other occurrence of that
    token, on any other branch, is loss-masked; that per-branch token loss mask is
    what stops the replicated trunk from contributing its gradient K+1 times.
    """
    seq = rollout.branch(k)
    index = {phys: pos for pos, phys in enumerate(seq)}
    rows, targets, physical = [], [], []
    for t in rollout.layer_targets(k):
        rows.append(index[t] - 1)
        targets.append(rollout.tokens[t])
        physical.append(t)
    return rows, targets, physical


def logittree_loss(params, rollout: Rollout, cfg: Config) -> jnp.ndarray:
    """LogitTree (Section 4.1): K+1 forwards, one per branch, losses summed.

    Costs K+1 forward and K+1 backward passes. Attention never crosses legs
    because the legs are physically separate sequences, so neither time-travel
    nor stale-context leakage has a pathway.
    """
    total = jnp.zeros((), dtype=jnp.float64)
    for k in range(rollout.n_junctions + 1):
        rows, targets, _ = branch_readout(rollout, k)
        if not rows:
            continue
        logits = branch_logits(params, rollout, k, cfg)
        total = total + cross_entropy(logits[jnp.asarray(rows)], jnp.asarray(targets))
    return total


def pack_branches(rollout: Rollout):
    """Build the packed sequence, the additive 4D mask, the relative offsets and
    the rows and targets the loss reads out of.

    The mask is block diagonal over branches and causal inside each block, which
    is condition (i) of Proposition 4 written on the replicated layout; position
    ids restart at zero per branch, which is condition (ii).
    """
    branches = rollout.branches()
    tokens, positions, block_id, offsets = [], [], [], []
    for b, seq in enumerate(branches):
        offsets.append(len(tokens))
        for pos, phys in enumerate(seq):
            tokens.append(rollout.tokens[phys])
            positions.append(pos)
            block_id.append(b)

    pos = jnp.asarray(positions)
    blk = jnp.asarray(block_id)
    same = blk[:, None] == blk[None, :]
    causal = pos[:, None] >= pos[None, :]
    bias = jnp.where(same & causal, 0.0, NEG)
    delta = pos[None, :] - pos[:, None]

    rows, targets = [], []
    for k in range(len(branches)):
        r, tgt, _ = branch_readout(rollout, k)
        rows.extend(offsets[k] + x for x in r)
        targets.extend(tgt)
    return (jnp.asarray(tokens), bias, delta,
            jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(targets, dtype=jnp.int32))


def packed4d_loss(params, rollout: Rollout, cfg: Config) -> jnp.ndarray:
    """The packed 4D mask (Section 4.2 / App. G): one forward, one backward.

    Same gradient placement as LogitTree, one pass instead of K+1, at the price
    of a custom mask and a longer packed input.
    """
    tokens, bias, delta, rows, targets = pack_branches(rollout)
    logits = forward(params, tokens, bias, delta, cfg)
    return cross_entropy(logits[rows], targets)
