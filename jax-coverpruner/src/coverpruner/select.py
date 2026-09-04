"""Demand-weighted coverage selection: Eq. 6 and Algorithm 1 of arXiv:2609.03158.

F is monotone submodular with non-negative utilities and weights, so the greedy
selector of Nemhauser et al. (1978) returns a set with F(S) >= (1 - 1/e) F(OPT).
greedy_select is the reference implementation, written to read like Eq. 6.
lazy_greedy_select is the accelerated version and returns the same set.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from coverpruner.coverage import calibrated_utilities

__all__ = [
    "Selection",
    "marginal_gains",
    "greedy_select",
    "lazy_greedy_select",
    "select_tokens",
]


class Selection(NamedTuple):
    """What the selector returns.

    Attributes:
        indices: retained token ids in the original visual-token order, which is the
            order Algorithm 1's Decode step feeds to the frozen decoder.
        order: the same ids in the order greedy picked them, most valuable first.
        coverage: F(S) after each pick, so coverage[-1] is the achieved objective.
    """

    indices: jnp.ndarray
    order: jnp.ndarray
    coverage: jnp.ndarray


def marginal_gains(A, w, c):
    """Gain of every candidate k against the current cover c: the sum inside Eq. 6.

    gain_k = sum_i w_i * max(0, A_ik - c_i)

    Args:
        A: (n, n) coverage utilities.
        w: (n,) demand weights.
        c: (n,) current per-token covered value, c_i = max_{j in S} A_ij.

    Returns:
        (n,) gain per candidate, including the ones already in S (their gain is 0).
    """
    return jnp.matmul(w, jnp.maximum(0.0, A - c[:, None]), precision="highest")


def greedy_select(A, w, budget):
    """Reference greedy for Eq. 6. Picks k* = argmax_k sum_i w_i max(0, A_ik - c_i).

    Args:
        A: (n, n) coverage utilities, A >= 0.
        w: (n,) demand weights, w >= 0.
        budget: m, the retained token count.

    Returns:
        Selection. Ties in the argmax go to the lowest token index.

    c starts at 0 for every token, as Eq. 6 says, and updates to
    c_i <- max(c_i, A_ik*) after each pick, so a step is scored by what is still
    uncovered rather than by any static token score.
    """
    A, w, budget = _check_inputs(A, w, budget)
    n = A.shape[0]
    c = jnp.zeros(n, dtype=A.dtype)
    chosen = np.zeros(n, dtype=bool)
    order, running = [], []

    for _ in range(budget):
        gains = np.array(marginal_gains(A, w, c), copy=True)
        gains[chosen] = -np.inf
        k = int(np.argmax(gains))
        chosen[k] = True
        order.append(k)
        c = jnp.maximum(c, A[:, k])
        running.append(float(jnp.sum(w * c)))

    return _finish(order, running)


def lazy_greedy_select(A, w, budget):
    """Lazy (accelerated) greedy, Minoux (1978). Same selection as greedy_select.

    Submodularity means a candidate's gain never grows as S grows, so a gain
    computed at an earlier step is an upper bound now. Keep the bounds in a max-heap
    and only re-evaluate the top one; if it still beats the next bound, it is the
    argmax and the rest never get touched.

    Args:
        A: (n, n) coverage utilities, A >= 0.
        w: (n,) demand weights, w >= 0.
        budget: m, the retained token count.

    Returns:
        Selection, identical to greedy_select's.

    Heap entries are (-gain, index), so the pop order among equal gains is by lowest
    index and the acceptance test compares the whole tuple. That reproduces
    greedy_select's tie-breaking instead of leaving it to heap layout.
    """
    A, w, budget = _check_inputs(A, w, budget)
    n = A.shape[0]
    c = jnp.zeros(n, dtype=A.dtype)

    gains = np.asarray(marginal_gains(A, w, c))
    heap = [(-float(g), int(k)) for k, g in enumerate(gains)]
    heapq.heapify(heap)

    order, running = [], []
    for _ in range(budget):
        while True:
            _, k = heapq.heappop(heap)
            fresh = -float(jnp.dot(w, jnp.maximum(0.0, A[:, k] - c)))
            if not heap or (fresh, k) <= heap[0]:
                break
            heapq.heappush(heap, (fresh, k))
        order.append(k)
        c = jnp.maximum(c, A[:, k])
        running.append(float(jnp.sum(w * c)))

    return _finish(order, running)


def select_tokens(
    V,
    *,
    budget,
    r=None,
    tau=0.3,
    A=None,
    lazy=True,
):
    """Run the CoverPruner selector, steps 1 and 3 of Algorithm 1.

    Args:
        V: (n, d) projected visual tokens g(f_v(X_v)). Projector-output space, not
            vision-encoder space (Sec. 3.3).
        budget: m, the retained token count.
        r: (n,) raw relevance from the layer-0 probe, Eq. 4. None means unweighted
            coverage, which is the tau = 0 setting.
        tau: query-conditioning temperature, paper default 0.3.
        A: precomputed utilities, if you already built them; V is ignored then.
        lazy: use lazy_greedy_select. The plain greedy is the same answer, slower.

    Returns:
        Selection.
    """
    from coverpruner.relevance import demand_weights

    if A is None:
        A = calibrated_utilities(V)
    n = A.shape[0]
    w = jnp.ones(n, dtype=A.dtype) if r is None else demand_weights(r, tau)
    select = lazy_greedy_select if lazy else greedy_select
    return select(A, w, budget)


def _check_inputs(A, w, budget):
    A = jnp.asarray(A, dtype=jnp.result_type(float))
    w = jnp.asarray(w, dtype=A.dtype)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    if w.shape != (A.shape[0],):
        raise ValueError(f"w must be ({A.shape[0]},), got {w.shape}")
    budget = int(budget)
    if not 0 <= budget <= A.shape[0]:
        raise ValueError(f"budget must be in [0, {A.shape[0]}], got {budget}")
    return A, w, budget


def _finish(order, running):
    order = jnp.asarray(order, dtype=jnp.int32)
    return Selection(
        indices=jnp.sort(order),
        order=order,
        coverage=jnp.asarray(running, dtype=jnp.result_type(float)),
    )
