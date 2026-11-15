"""The four construction rules of Section 6.2 of arXiv:2609.04420.

Three of them are probability samples and estimate the same estimand L(Phi) = sum_i l_i
without bias (Proposition 4):

  uniform            K rows drawn from all n without replacement, weights w_j = n/K;
  class-stratified   K_c rows drawn without replacement inside class c, w_j = N_c/K_c,
                     with K_c = K N_c / n (proportional allocation);
  class-balanced     the same estimator with K_c set by the truncation
                     K_c = min{N_c, floor(K/2)} of Section 6.4.

The fourth, K-means representatives, is not a probability sample and lives in
``representatives.py`` with the bias identity of Proposition 7 that governs it.

Sampling without replacement is done by taking the first K entries of a random
permutation, which is exactly simple random sampling without replacement.
"""

from typing import Tuple

import jax
import jax.numpy as jnp


def class_index_sets(labels: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Row indices of class 0 and class 1, as two arrays with static shapes."""
    labels = jnp.asarray(labels)
    order = jnp.argsort(labels, stable=True)
    n0 = int(jnp.sum(labels == 0))
    return order[:n0], order[n0:]


def proportional_allocation(
    counts: jax.Array, budget: int, min_per_class: int = 1
) -> Tuple[int, int]:
    """K_c = K N_c / n, the class-stratified rule of Section 6.2.

    Proposition 6 and Equation (9) assume this is exactly integral, and it usually is not
    in the rare-event regime; the leftover row goes to the class with the larger
    fractional part. Equation (8) also needs 0 < K_c, so at rare prevalence the exact
    share for class 1 rounds to zero and ``min_per_class`` pulls it back to one. Both
    adjustments break Equation (9), so pick n, K and N_1 with K N_c / n integral when you
    want to check that identity.
    """
    n0, n1 = float(counts[0]), float(counts[1])
    n = n0 + n1
    exact = (budget * n0 / n, budget * n1 / n)
    k = [int(exact[0]), int(exact[1])]
    if k[0] + k[1] < budget:
        k[0 if (exact[0] - k[0]) >= (exact[1] - k[1]) else 1] += 1
    k[0] = max(k[0], min_per_class)
    k[1] = max(k[1], min_per_class)
    if k[0] + k[1] > budget:
        k[0 if k[0] > k[1] else 1] -= k[0] + k[1] - budget
    return _clip(k[0], k[1], n0, n1, budget)


def balanced_allocation(counts: jax.Array, budget: int, floor_at: int = 0) -> Tuple[int, int]:
    """K_c = min{N_c, max(floor_at, floor(K/2))}, Section 6.4.

    Corollary 12 ignores the floor at 25 that the paper's implementation carries, so
    ``floor_at`` defaults to 0 and the rule is the plain truncation K_c = min{N_c,
    floor(K/2)}. When positives are scarce the design keeps all of them and the realised
    total falls short of K; that shortfall is the whole content of the realised ratio
    gamma = min{2 pi / f, 1} of Equation (17).
    """
    half = max(floor_at, budget // 2)
    return int(min(counts[0], half)), int(min(counts[1], half))


def neyman_allocation(
    counts: jax.Array, dispersions: jax.Array, budget: int, integer: bool = False
):
    """K_c* = K N_c S_c / (N_0 S_0 + N_1 S_1), Equation (11) of Theorem 8.

    With ``integer=True`` the continuous optimum is rounded and the two counts are made
    to sum to the budget. The objective sum_c N_c^2 S_c^2 / K_c is convex in K_0, so the
    integer minimiser is one of the two neighbours of the continuous one.
    """
    weight = counts * dispersions
    exact = budget * weight / jnp.sum(weight)
    if not integer:
        return exact
    k0 = int(jnp.round(exact[0]))
    k0 = min(max(k0, 1), budget - 1)
    return _clip(k0, budget - k0, float(counts[0]), float(counts[1]), budget)


def _clip(k0: int, k1: int, n0: float, n1: float, budget: int) -> Tuple[int, int]:
    k0 = min(k0, int(n0))
    k1 = min(k1, int(n1))
    spare = budget - k0 - k1
    if spare > 0:
        take = min(spare, int(n0) - k0)
        k0 += take
        spare -= take
        k1 += min(spare, int(n1) - k1)
    return int(k0), int(k1)


def uniform_sample(key: jax.Array, n: int, budget: int) -> jax.Array:
    """K of n rows without replacement."""
    return jax.random.permutation(key, n)[:budget]


def uniform_estimate(key: jax.Array, losses: jax.Array, budget: int) -> jax.Array:
    """Lhat = (n/K) sum_{j in S} l_j, the uniform design of Section 6.2."""
    n = losses.shape[0]
    idx = uniform_sample(key, n, budget)
    return (n / budget) * jnp.sum(losses[idx])


def stratified_sample(
    key: jax.Array, idx0: jax.Array, idx1: jax.Array, k0: int, k1: int
) -> Tuple[jax.Array, jax.Array]:
    """K_c rows without replacement inside each class, drawn independently."""
    key0, key1 = jax.random.split(key)
    s0 = idx0[jax.random.permutation(key0, idx0.shape[0])[:k0]]
    s1 = idx1[jax.random.permutation(key1, idx1.shape[0])[:k1]]
    return s0, s1


def stratified_estimate(
    key: jax.Array,
    losses: jax.Array,
    idx0: jax.Array,
    idx1: jax.Array,
    k0: int,
    k1: int,
) -> jax.Array:
    """Lhat = sum_c (N_c/K_c) sum_{j in S_c} l_j, the class-conditional designs."""
    s0, s1 = stratified_sample(key, idx0, idx1, k0, k1)
    n0, n1 = idx0.shape[0], idx1.shape[0]
    return (n0 / k0) * jnp.sum(losses[s0]) + (n1 / k1) * jnp.sum(losses[s1])
