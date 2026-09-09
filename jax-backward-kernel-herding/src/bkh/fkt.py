"""Flexible Kernel Thinning: bit extraction and tree-pruning (Algorithm 5).

Dyadic expansion of p (eqs 6-8):
    p = sum_i b_i / 2^i ,   b_i in {0, 1}
    b_i = 1[r_{i-1} >= 1/2^i],  r_i = r_{i-1} - b_i / 2^i,  r_0 = p
    |S_FKT| = sum_i b_i floor(n / 2^i)  ≈ p n

Collection: Algorithm 5's foreach-active-node step would add one child per
active node when b_i = 1, which over-counts whenever an earlier bit is zero
(several active nodes at that level). Equation (8) asks for floor(n / 2^i)
points in total at bit i. This module therefore builds the KT-SPLIT tree
(Algorithm 3) and, at each b_i = 1, adds unused level-i nodes until
floor(n / 2^i) points are taken, choosing the child that yields smaller
MMD_{k_rt}^2 to S_n as in Algorithm 5 line 14.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from bkh.kernels import gaussian_kernel, gaussian_sqrt_kernel, mmd2_indices


def dyadic_bits(p: float, tau: float = 1e-6, max_depth: int = 40):
    """Greedy dyadic expansion of p (Algorithm 5 Step 1, eq 7).

    Returns:
        bits: tuple of 0/1 of length L, index 0 is b_1.
        residual: leftover r_L = p - sum_i b_i / 2^i, with residual < tau
            (or after max_depth).
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    r = float(p)
    bits = []
    depth = 0
    while r >= tau and depth < max_depth:
        depth += 1
        thresh = 2.0 ** (-depth)
        if r >= thresh:
            bits.append(1)
            r = r - thresh
        else:
            bits.append(0)
    return tuple(bits), r


def reconstructed_p(bits) -> float:
    """sum_i b_i / 2^i from a bit tuple whose first entry is b_1."""
    total = 0.0
    for i, b in enumerate(bits, start=1):
        if b:
            total += 2.0 ** (-i)
    return total


def coreset_size_from_bits(n: int, bits) -> int:
    """Equation (8): sum_i b_i floor(n / 2^i)."""
    size = 0
    for i, b in enumerate(bits, start=1):
        if b:
            size += n // (2**i)
    return size


def _halve_indices(idx, k_rt, key, delta: float = 0.5):
    """One KT-SPLIT halving of a parent index sequence (Algorithm 3).

    Pairs consecutive points and assigns them with a compact self-balancing
    walk (eq 9): swap probability uses the running RKHS imbalance against
    k_rt(x, ·) - k_rt(y, ·). Odd leftover points are assigned at random.
    """
    idx = jnp.asarray(idx, dtype=jnp.int32)
    n = int(idx.size)
    if n == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty
    if n == 1:
        key, sub = jax.random.split(key)
        go_left = bool(jax.random.bernoulli(sub, 0.5))
        empty = jnp.zeros((0,), dtype=jnp.int32)
        if go_left:
            return idx, empty
        return empty, idx

    n_pairs = n // 2
    left = []
    right = []
    # Running inner products of the signed walk with k_rt(x_a, ·).
    walk = jnp.zeros((k_rt.shape[0],), dtype=k_rt.dtype)
    log_mult = 2.0 * math.log(max(2.0 * n / max(delta, 1e-12), 2.0))
    sig_sqd = 0.0

    for t in range(n_pairs):
        key, sub = jax.random.split(key)
        i = int(idx[2 * t])
        j = int(idx[2 * t + 1])
        b2 = float(k_rt[i, i] + k_rt[j, j] - 2.0 * k_rt[i, j])
        b2 = max(b2, 0.0)
        # Delta_t: imbalance inner product with (k_rt(x_i) - k_rt(x_j)).
        delta_t = float(walk[i] - walk[j])
        thresh = max(math.sqrt(max(sig_sqd, 0.0) * b2 * log_mult), b2, 1e-12)
        if sig_sqd == 0.0:
            sig_sqd = b2
        else:
            update = 0.5 + (b2 / (2.0 * thresh) - 1.0) * sig_sqd / thresh
            if update > 0.0:
                sig_sqd = sig_sqd + 2.0 * b2 * update
        # Eq (9): delta_t = min(1, max(0, 1/2 (1 - Delta_t / a_t))).
        prob = 0.5 * (1.0 - delta_t / thresh)
        prob = min(1.0, max(0.0, prob))
        # sigma = +1 sends x_i to left, x_j to right (Algorithm 3).
        sigma_plus = bool(jax.random.uniform(sub) > prob)
        if sigma_plus:
            left.append(i)
            right.append(j)
            sign = 1.0
        else:
            left.append(j)
            right.append(i)
            sign = -1.0
        walk = walk + sign * (k_rt[:, i] - k_rt[:, j])

    if n % 2 == 1:
        key, sub = jax.random.split(key)
        leftover = int(idx[-1])
        if bool(jax.random.bernoulli(sub, 0.5)):
            left.append(leftover)
        else:
            right.append(leftover)

    left_arr = jnp.asarray(left, dtype=jnp.int32) if left else jnp.zeros((0,), dtype=jnp.int32)
    right_arr = jnp.asarray(right, dtype=jnp.int32) if right else jnp.zeros((0,), dtype=jnp.int32)
    return left_arr, right_arr


def kt_split_tree(k_rt, depth: int, key, delta: float = 0.5):
    """KT-SPLIT tree of intermediate index sets (Algorithm 3 / Alg 5 Step 2).

    nodes[j] is a list of 2^j index arrays S^{(j, ell)}.
    """
    n = k_rt.shape[0]
    nodes = [[jnp.arange(n, dtype=jnp.int32)]]
    for _j in range(1, depth + 1):
        children = []
        for parent in nodes[-1]:
            key, sub = jax.random.split(key)
            left, right = _halve_indices(parent, k_rt, sub, delta=delta)
            children.append(left)
            children.append(right)
        nodes.append(children)
    return nodes


def _node_unused(node, used):
    node = jnp.asarray(node)
    if node.size == 0:
        return False
    return not bool(jnp.any(used[node]))


def _best_child(candidates, selected, used, k_rt, all_idx):
    """Choose unused candidate minimizing MMD_{k_rt}^2(S ∪ ν, S_n)."""
    best = None
    best_val = None
    for node in candidates:
        node = jnp.asarray(node)
        if node.size == 0 or not _node_unused(node, used):
            continue
        if selected.size == 0:
            union = node
        else:
            union = jnp.concatenate([selected, node])
        val = float(mmd2_indices(k_rt, union, all_idx))
        if best_val is None or val < best_val:
            best_val = val
            best = node
    return best


def tree_prune_collect(nodes, bits, k_rt):
    """Tree-pruning collection (Algorithm 5 Step 3, sized by eq 8).

    For each b_i = 1, add unused level-i nodes until floor(n / 2^i) points
    are taken, picking the child with smaller MMD_{k_rt} to S_n.
    """
    n = int(k_rt.shape[0])
    all_idx = jnp.arange(n, dtype=jnp.int32)
    used = jnp.zeros((n,), dtype=bool)
    selected = jnp.zeros((0,), dtype=jnp.int32)
    max_level = len(nodes) - 1

    for i, bit in enumerate(bits, start=1):
        if not bit:
            continue
        need = n // (2**i)
        if need <= 0 or i > max_level:
            continue
        taken = 0
        while taken < need:
            node = _best_child(nodes[i], selected, used, k_rt, all_idx)
            if node is None:
                break
            selected = jnp.concatenate([selected, node])
            used = used.at[node].set(True)
            taken += int(node.size)
    return selected


def flexible_kernel_thinning(
    x,
    p: float,
    tau: float = 1e-6,
    sigma: float = 1.0,
    key=None,
    swap: bool = False,
    delta: float = 0.5,
):
    """Flexible Kernel Thinning without the cubic KT-SWAP by default.

    Steps 1-3 of Algorithm 5 (bit extraction, KT-SPLIT tree, collection).
    KT-SWAP (Step 4) is optional and omitted unless swap=True, since tests
    only need split/collection and the square-root kernel.

    Returns:
        coreset indices (int32 array) of size sum_i b_i floor(n / 2^i).
        bits: the dyadic bit tuple.
    """
    x = jnp.asarray(x)
    n = x.shape[0]
    bits, _resid = dyadic_bits(p, tau=tau)
    depth = len(bits)
    # Splitting below one point per node cannot add mass under eq (8).
    max_useful = max(int(math.floor(math.log2(max(n, 1)))), 0)
    depth = min(depth, max_useful)
    bits = bits[:depth] if depth else bits

    if key is None:
        key = jax.random.PRNGKey(0)
    k_rt = gaussian_sqrt_kernel(x, x, sigma=sigma)
    nodes = kt_split_tree(k_rt, depth, key, delta=delta)
    coreset = tree_prune_collect(nodes, bits, k_rt)
    if swap and coreset.size > 0:
        k = gaussian_kernel(x, x, sigma=sigma)
        coreset = _tiny_kt_swap(k, coreset)
    return coreset, bits


def _tiny_kt_swap(k, coreset, max_passes: int = 1):
    """One greedy swap pass (Algorithm 4 style) on a tiny coreset.

    For each coreset index, try replacing it with an unused point if that
    strictly lowers MMD^2 to the full set. Cardinality is unchanged.
    """
    n = k.shape[0]
    all_idx = jnp.arange(n, dtype=jnp.int32)
    core = list(int(i) for i in coreset)
    in_core = set(core)
    for _ in range(max_passes):
        improved = False
        for pos, old in enumerate(list(core)):
            best = old
            best_val = float(mmd2_indices(k, jnp.asarray(core, dtype=jnp.int32), all_idx))
            for cand in range(n):
                if cand in in_core:
                    continue
                trial = list(core)
                trial[pos] = cand
                val = float(mmd2_indices(k, jnp.asarray(trial, dtype=jnp.int32), all_idx))
                if val < best_val:
                    best_val = val
                    best = cand
            if best != old:
                in_core.remove(old)
                in_core.add(best)
                core[pos] = best
                improved = True
        if not improved:
            break
    return jnp.asarray(core, dtype=jnp.int32)
