"""Squared-Euclidean Gromov-Wasserstein objective and nested solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gwsgp.sinkhorn import sinkhorn


def gw_cost_matrix(plan: jax.Array, adj_a: jax.Array, adj_b: jax.Array) -> jax.Array:
    """Linearized GW cost ``C`` with elementwise-squared adjacencies.

    ``C = (A ⊙ A) 1 1^T + 1 1^T (B ⊙ B)^T - 2 A T B^T``

    so that ``J(T; A, B) = <C, T> = sum_{ijkl} (A_ik - B_jl)^2 T_ij T_kl``.
    """
    a_sq = jnp.sum(adj_a * adj_a, axis=-1, keepdims=True)
    b_sq = jnp.sum(adj_b * adj_b, axis=-1, keepdims=True)
    atb = adj_a @ plan @ jnp.swapaxes(adj_b, -1, -2)
    return a_sq + jnp.swapaxes(b_sq, -1, -2) - 2.0 * atb


def gw_loss(plan: jax.Array, adj_a: jax.Array, adj_b: jax.Array) -> jax.Array:
    """Quadratic matching cost ``J(T; A, B)``.

    Returns a scalar for unbatched ``(n, n)`` inputs, or a vector of shape
    ``batch`` when the leading axes are batched.
    """
    cost = gw_cost_matrix(plan, adj_a, adj_b)
    return jnp.sum(cost * plan, axis=(-2, -1))


def gw_grad_plan(plan: jax.Array, adj_a: jax.Array, adj_b: jax.Array) -> jax.Array:
    """Analytic gradient of ``J`` with respect to ``T``."""
    cost = gw_cost_matrix(plan, adj_a, adj_b)
    a_t = jnp.swapaxes(adj_a, -1, -2)
    return cost - 2.0 * (a_t @ plan @ adj_b)


def nested_gw(
    adj_a: jax.Array,
    adj_b: jax.Array,
    tau: float = 0.08,
    n_outer: int = 20,
    n_inner: int = 24,
) -> jax.Array:
    """Mirror-descent GW solver (Algorithm 2): Sinkhorn-in-the-loop.

    Each outer step linearizes ``J`` at the current plan and solves an
    entropic OT problem with temperature ``tau``.
    """
    deg_a = jnp.sum(adj_a, axis=-1)
    deg_b = jnp.sum(adj_b, axis=-1)
    node_cost = (deg_a[..., :, None] - deg_b[..., None, :]) ** 2
    plan = sinkhorn(node_cost, epsilon=max(tau, 0.1), n_iters=n_inner)

    def step(plan, _):
        cost = gw_cost_matrix(plan, adj_a, adj_b)
        plan = sinkhorn(cost, epsilon=tau, n_iters=n_inner)
        return plan, None

    plan, _ = jax.lax.scan(step, plan, None, length=n_outer)
    return plan


def greedy_permutation(plan) -> jax.Array:
    """Greedy argmax rounding of a coupling to a permutation (row -> col)."""
    plan_np = jnp.asarray(plan)
    n = plan_np.shape[0]
    scores = plan_np.reshape(-1)
    order = jnp.argsort(-scores)
    assigned_row = jnp.zeros((n,), dtype=bool)
    assigned_col = jnp.zeros((n,), dtype=bool)
    perm = jnp.zeros((n,), dtype=jnp.int32)

    def body(carry, idx):
        assigned_row, assigned_col, perm = carry
        i = idx // n
        j = idx % n
        free = (~assigned_row[i]) & (~assigned_col[j])
        assigned_row = assigned_row.at[i].set(assigned_row[i] | free)
        assigned_col = assigned_col.at[j].set(assigned_col[j] | free)
        perm = perm.at[i].set(jnp.where(free, j, perm[i]))
        return (assigned_row, assigned_col, perm), None

    (_, _, perm), _ = jax.lax.scan(body, (assigned_row, assigned_col, perm), order)
    return perm


def permutation_matrix(perm: jax.Array, n: int | None = None) -> jax.Array:
    """One-hot permutation matrix with ``P[i, perm[i]] = 1``."""
    n = int(perm.shape[0] if n is None else n)
    return jax.nn.one_hot(perm, n, dtype=jnp.float32)


def hamming_perm(pred: jax.Array, true: jax.Array) -> jax.Array:
    return jnp.sum(pred != true)
