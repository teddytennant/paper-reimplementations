import jax
import jax.numpy as jnp
import numpy as np

from gwsgp.data import complete_adjacency, cycle_adjacency, path_adjacency
from gwsgp.gw import greedy_permutation, gw_grad_plan, gw_loss, nested_gw, permutation_matrix
from gwsgp.sinkhorn import sinkhorn


def test_j_identity_zero():
    adj = cycle_adjacency(6)
    identity = jnp.eye(6)
    value = gw_loss(identity, adj, adj)
    assert jnp.allclose(value, 0.0, atol=1e-5)


def test_j_true_permutation_zero():
    adj = path_adjacency(6)
    perm = jnp.array([2, 0, 5, 1, 4, 3], dtype=jnp.int32)
    plan = permutation_matrix(perm)
    # Tensor GW vanishes at T=P when B = P^T A P (relabel nodes of A by P).
    adj_b = plan.T @ adj @ plan
    value = gw_loss(plan, adj, adj_b)
    assert jnp.allclose(value, 0.0, atol=1e-5)


def test_j_nonisomorphic_larger_than_isomorphic():
    cycle = cycle_adjacency(6)
    complete = complete_adjacency(6)
    perm = jnp.array([1, 5, 2, 0, 4, 3], dtype=jnp.int32)
    plan = permutation_matrix(perm)
    cycle_perm = plan.T @ cycle @ plan
    j_iso = gw_loss(plan, cycle, cycle_perm)
    plan_diff = nested_gw(cycle, complete, tau=0.1, n_outer=12, n_inner=16)
    j_diff = gw_loss(plan_diff, cycle, complete)
    assert j_iso < 1e-4
    assert j_diff > j_iso + 1.0


def test_nested_gw_decreases_j():
    adj = path_adjacency(5)
    perm = jnp.array([4, 1, 0, 3, 2], dtype=jnp.int32)
    plan_true = permutation_matrix(perm)
    adj_b = plan_true.T @ adj @ plan_true
    uniform = jnp.full((5, 5), 1.0 / 5.0)
    j0 = gw_loss(uniform, adj, adj_b)
    plan = nested_gw(adj, adj_b, tau=0.08, n_outer=15, n_inner=20)
    j1 = gw_loss(plan, adj, adj_b)
    assert j1 < j0


def _path_automorphisms(n: int):
    identity = np.arange(n)
    reverse = np.arange(n - 1, -1, -1)
    return identity, reverse


def test_nested_gw_recovers_permuted_path():
    n = 5
    adj = path_adjacency(n)
    perm = jnp.array([2, 4, 1, 0, 3], dtype=jnp.int32)
    true_plan = permutation_matrix(perm)
    adj_b = true_plan.T @ adj @ true_plan
    plan = nested_gw(adj, adj_b, tau=0.05, n_outer=30, n_inner=24)
    pred = np.asarray(greedy_permutation(plan))
    true = np.asarray(perm)
    hammings = []
    for auto in _path_automorphisms(n):
        composed = true[auto]
        hammings.append(int(np.sum(pred != composed)))
    assert min(hammings) <= 2


def test_gw_grad_through_sinkhorn_is_finite():
    adj_a = cycle_adjacency(5)
    adj_b = complete_adjacency(5)

    def objective(adj):
        cost = jnp.sum((adj[:, None, :] - adj_b[None, :, :]) ** 2, axis=-1)
        plan = sinkhorn(cost, epsilon=0.2, n_iters=12)
        return gw_loss(plan, adj, adj_b)

    grad = jax.grad(objective)(adj_a)
    assert jnp.all(jnp.isfinite(grad))


def test_analytic_gw_grad_matches_jax():
    key = jax.random.PRNGKey(3)
    adj_a = cycle_adjacency(4)
    adj_b = path_adjacency(4)
    plan = jax.nn.softmax(jax.random.normal(key, (4, 4)), axis=1)
    plan = plan / jnp.sum(plan, axis=0, keepdims=True) * 4.0 / 4.0
    analytic = gw_grad_plan(plan, adj_a, adj_b)
    autodiff = jax.grad(lambda t: gw_loss(t, adj_a, adj_b))(plan)
    assert jnp.allclose(analytic, autodiff, atol=1e-4)
