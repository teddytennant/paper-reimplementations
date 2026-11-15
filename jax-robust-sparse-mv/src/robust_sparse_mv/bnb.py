"""Enumeration branch-and-bound for (P1) with the Proposition 11 pruning rule."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from robust_sparse_mv.closed_form import (
    Xi,
    dual_multiplier,
    norm_d,
    submatrix,
    xi,
)
from robust_sparse_mv.problem import RobustSparseMV


@dataclass
class Node:
    x: Array
    lb: float
    ub: float
    P: tuple[int, ...]
    S: tuple[int, ...]


@dataclass
class BnBResult:
    x: Array
    f_beta: float
    support: tuple[int, ...]
    n_xi_evals: int
    n_nodes: int
    n_right_pruned: int
    n_enqueued: int
    history: list[str] = field(default_factory=list)


def _sorted_tuple(indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(i) for i in indices))


def _quadratic_on_support(problem: RobustSparseMV, omega: Sequence[int], w: Array) -> float:
    d_w = submatrix(problem.D, omega)
    return float(w @ (d_w @ w))


def _branching_index(problem: RobustSparseMV, P: tuple[int, ...], S: tuple[int, ...]) -> int:
    """Section 5.2: Lagrangian gradient if ``S`` nonempty, else variance-to-return ratio."""
    p = np.asarray(P, dtype=np.int32)
    if len(S) >= 1:
        s = np.asarray(S, dtype=np.int32)
        w = xi(problem, s)
        mu = float(dual_multiplier(problem, s))
        d_ps = problem.D[jnp.ix_(p, s)]
        r_p = problem.r[p]
        nrm = float(norm_d(w, submatrix(problem.D, s)))
        g = 2.0 * (d_ps @ w) - mu * (r_p - problem.gamma * (d_ps @ w) / nrm)
        g_abs = np.abs(np.asarray(g, dtype=np.float64))
        return int(p[int(np.argmax(g_abs))])
    # S empty: j ∈ arg min_{i∈P} diag(D_P)[i] / r_P[i]
    diag = np.asarray(jnp.diag(problem.D), dtype=np.float64)[p]
    r_p = np.asarray(problem.r, dtype=np.float64)[p]
    ratios = diag / r_p
    return int(p[int(np.argmin(ratios))])


def branch_and_bound_p1(
    problem: RobustSparseMV,
    *,
    eps: float = 1e-8,
    extra_pruning: bool = True,
    max_nodes: int = 100_000,
) -> BnBResult:
    """Solve (P1) by the enumeration BnB of Algorithm 1 (no warm-start).

    Nodes are ``(x, lb, ub, P, S)`` with a min-``lb`` priority queue and FIFO
    tie-break. The extra pruning rule (Proposition 11) skips the right child
    whenever ``lb^R + β ≥ ub* − ε``.
    """
    n = problem.n
    full = tuple(range(n))
    n_xi = 0

    def eval_xi(omega: Sequence[int]) -> Array:
        nonlocal n_xi
        n_xi += 1
        return xi(problem, omega)

    u_full = eval_xi(full)
    root_lb = _quadratic_on_support(problem, full, u_full)
    x_full = Xi(problem, full)

    ub_star = float("inf")
    x_star = x_full
    support_star: tuple[int, ...] = full

    seq = 0
    heap: list[tuple[float, int, Node]] = []

    def enqueue(node: Node) -> None:
        nonlocal seq
        heapq.heappush(heap, (node.lb, seq, node))
        seq += 1

    enqueue(Node(x=x_full, lb=root_lb, ub=float("inf"), P=full, S=()))
    n_enqueued = 1
    n_nodes = 0
    n_right_pruned = 0
    history: list[str] = []

    while heap and n_nodes < max_nodes:
        _prio, _seq, node = heapq.heappop(heap)
        n_nodes += 1
        if node.ub < ub_star:
            ub_star = float(node.ub)
            x_star = node.x
            support_star = node.S
            history.append(f"incumbent S={support_star} ub={ub_star:.12g}")
        if ub_star - node.lb <= eps:
            break
        if not node.P:
            continue

        j = _branching_index(problem, node.P, node.S)
        p_child = _sorted_tuple(i for i in node.P if i != j)
        s_left = _sorted_tuple((*node.S, j))
        s_right = _sorted_tuple((*node.S, *p_child))

        # Right child: P^R = P \ {j}, S^R = S ∪ P^R, ub^R = parent ub.
        if len(s_right) >= 1:
            w_r = eval_xi(s_right)
            lb_r = _quadratic_on_support(problem, s_right, w_r) + problem.beta * len(node.S)
            prune = extra_pruning and (lb_r + problem.beta >= ub_star - eps)
            if prune:
                n_right_pruned += 1
                history.append(
                    f"prune-right S={node.S} j={j} lbR={lb_r:.12g} ub*={ub_star:.12g}"
                )
            else:
                enqueue(
                    Node(
                        x=node.x,
                        lb=lb_r,
                        ub=node.ub,
                        P=p_child,
                        S=node.S,
                    )
                )
                n_enqueued += 1

        # Left child: S^L = S ∪ {j}, P^L = P \ {j}.
        if len(s_left) >= 1:
            w_l = eval_xi(s_left)
            ub_l = _quadratic_on_support(problem, s_left, w_l) + problem.beta * len(s_left)
            x_l = jnp.zeros((n,), dtype=problem.D.dtype).at[jnp.asarray(s_left)].set(w_l)
            enqueue(
                Node(
                    x=x_l,
                    lb=node.lb + problem.beta,
                    ub=ub_l,
                    P=p_child,
                    S=s_left,
                )
            )
            n_enqueued += 1

    return BnBResult(
        x=x_star,
        f_beta=float(ub_star),
        support=support_star,
        n_xi_evals=n_xi,
        n_nodes=n_nodes,
        n_right_pruned=n_right_pruned,
        n_enqueued=n_enqueued,
        history=history,
    )
