"""SDCC: self-distillation for conditioning consistency (Section 5).

One gradient-carrying forward on the compressed walk is the student. The live
prefix of each diverging leaf is rebuilt by re-inserting the spans active there
(Eq. 9) and evaluated with gradients stopped; that is the teacher. A forward KL
at those leaves, and nowhere else, pulls the student's next-token distribution
back onto the one the policy actually had at rollout time (Eq. 10):

    L_SDCC = L_task(theta; H_comp)
             + lambda * sum_p KL( P_target(. | p_t(y_p)) || P_theta(. | p_s(y_p))_p ).

"Teacher" is the current policy with a stop-gradient, not a second model. The KL
is forward, so the teacher is a fixed soft target and the student is pushed to
cover what was likely under the live context.

Cost. The teacher legs are gradient-free and every diverging leaf in branch
layer k reads out of the same layer-k forward, so a rollout needs one backward
pass and at most K extra no-grad forwards; layer K re-converges and has no
diverging leaves. Setting ``lam=0`` recovers Naive-Compressed exactly, and if no
context differs the leaf set is empty and the loss is the plain task loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .exact import branch_logits
from .model import Config, sequence_logits
from .naive import naive_compressed_loss
from .tree import Rollout


def teacher_prefix(rollout: Rollout, p: int) -> tuple[int, ...]:
    """Eq. 9: trunk, re-inserted spans, post-junction suffix.

    Built by insertion rather than by calling ``live_view`` so the construction
    the paper writes is the one being tested; the assertion states that the two
    agree, including the overlapping-eviction case of App. D.1.
    """
    prefix = list(rollout.compressed_prefix(p))
    for e in rollout.active_evictions(p):
        for pos in sorted(x for x in e.span if x < p):
            slot = sum(1 for q in prefix if q < pos)  # iota(E_k), the original slot
            prefix.insert(slot, pos)
    assert tuple(prefix) == rollout.live_view(p)
    return tuple(prefix)


def student_rows(rollout: Rollout) -> tuple[list[int], list[int]]:
    """Rows of the compressed-walk forward that read out the diverging leaves."""
    walk = rollout.compressed_walk()
    index = {phys: pos for pos, phys in enumerate(walk)}
    leaves = rollout.diverging_leaves()
    return [index[p] - 1 for p in leaves], list(leaves)


def teacher_logits(params, rollout: Rollout, cfg: Config) -> dict[int, jnp.ndarray]:
    """Stop-gradient teacher distributions at every diverging leaf.

    Grouped by branch layer: one no-grad forward per layer that holds a leaf.
    """
    frozen = jax.lax.stop_gradient(params)
    leaves = rollout.diverging_leaves()
    by_layer: dict[int, list[int]] = {}
    for p in leaves:
        by_layer.setdefault(rollout.layer_of(p), []).append(p)

    out: dict[int, jnp.ndarray] = {}
    for k, ps in by_layer.items():
        logits = branch_logits(frozen, rollout, k, cfg)
        seq = rollout.branch(k)
        index = {phys: pos for pos, phys in enumerate(seq)}
        for p in ps:
            out[p] = logits[index[p] - 1]
    return out


def teacher_logits_by_reconstruction(params, rollout: Rollout, cfg: Config) -> dict[int, jnp.ndarray]:
    """Same teachers, one forward per leaf over the Eq. 9 reconstruction."""
    frozen = jax.lax.stop_gradient(params)
    out = {}
    for p in rollout.diverging_leaves():
        ids = jnp.asarray([rollout.tokens[i] for i in teacher_prefix(rollout, p)])
        out[p] = sequence_logits(frozen, ids, cfg)[-1]
    return out


def forward_kl(teacher_logits_p: jnp.ndarray, student_logits_p: jnp.ndarray) -> jnp.ndarray:
    """``KL(teacher || student)`` between two next-token distributions."""
    lt = jax.nn.log_softmax(teacher_logits_p)
    ls = jax.nn.log_softmax(student_logits_p)
    return jnp.sum(jnp.exp(lt) * (lt - ls))


def total_variation(logits_a: jnp.ndarray, logits_b: jnp.ndarray) -> jnp.ndarray:
    """``||P_a - P_b||_TV``, half the L1 distance."""
    pa = jax.nn.softmax(logits_a)
    pb = jax.nn.softmax(logits_b)
    return 0.5 * jnp.sum(jnp.abs(pa - pb))


def residual_kl(params, rollout: Rollout, cfg: Config) -> dict[int, jnp.ndarray]:
    """``eps_p`` of Section 5.2, the KL left at each diverging leaf."""
    rows, leaves = student_rows(rollout)
    if not leaves:
        return {}
    walk = rollout.compressed_walk()
    ids = jnp.asarray([rollout.tokens[i] for i in walk])
    student = sequence_logits(params, ids, cfg)
    teachers = teacher_logits(params, rollout, cfg)
    return {p: forward_kl(teachers[p], student[r]) for r, p in zip(rows, leaves)}


def leaf_gap(params, rollout: Rollout, cfg: Config) -> dict[int, tuple[jnp.ndarray, jnp.ndarray]]:
    """Per-leaf ``(eps_p, TV)`` for checking Proposition 2 directly."""
    rows, leaves = student_rows(rollout)
    if not leaves:
        return {}
    walk = rollout.compressed_walk()
    ids = jnp.asarray([rollout.tokens[i] for i in walk])
    student = sequence_logits(params, ids, cfg)
    teachers = teacher_logits(params, rollout, cfg)
    return {
        p: (forward_kl(teachers[p], student[r]), total_variation(teachers[p], student[r]))
        for r, p in zip(rows, leaves)
    }


def sdcc_loss(params, rollout: Rollout, cfg: Config, lam: float = 1.0) -> jnp.ndarray:
    """Eq. 10. ``lam=0`` gives Naive-Compressed back, bit for bit."""
    task = naive_compressed_loss(params, rollout, cfg)
    rows, leaves = student_rows(rollout)
    if not leaves or lam == 0.0:
        return task
    walk = rollout.compressed_walk()
    ids = jnp.asarray([rollout.tokens[i] for i in walk])
    student = sequence_logits(params, ids, cfg)
    teachers = teacher_logits(params, rollout, cfg)
    penalty = jnp.zeros((), dtype=jnp.float64)
    for r, p in zip(rows, leaves):
        penalty = penalty + forward_kl(teachers[p], student[r])
    return task + lam * penalty
