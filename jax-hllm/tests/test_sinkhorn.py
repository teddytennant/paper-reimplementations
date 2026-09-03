"""Equation (2) has to reach the Birkhoff polytope, and equation (3) has to
have the gradient it claims."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hllm.hungarian import decode_ranking
from hllm.sinkhorn import log_sinkhorn, sinkhorn, sinkhorn_loss


def naive_sinkhorn(scores, tau, n_iters):
    """Equation (2) transcribed literally, in probability space, for reference.

    Only usable at temperatures where ``exp(M / tau)`` survives float64.
    """
    s = np.exp(np.asarray(scores, dtype=np.float64) / tau)
    for _ in range(n_iters):
        s = s / s.sum(axis=0, keepdims=True)
        s = s / s.sum(axis=1, keepdims=True)
    return s


@pytest.mark.parametrize("n", [3, 8, 25])
@pytest.mark.parametrize(
    "tau,n_iters,atol", [(1.0, 200, 1e-12), (0.3, 3000, 1e-6), (0.05, 3000, 2e-2)]
)
def test_plan_is_doubly_stochastic(n, tau, n_iters, atol):
    """Row and column sums both go to 1, which is what "lands on the Birkhoff
    polytope" means in Section 3.3.

    The budget and the tolerance move together on purpose: the entropic problem
    stiffens as tau falls, and at tau = 0.05 three thousand passes still leave
    the columns off by about 1e-3. Rows are exact at any budget because
    :func:`log_sinkhorn` ends on the row pass.
    """
    rng = np.random.default_rng(n)
    scores = rng.normal(size=(n, n)) * 2.0
    plan = np.asarray(sinkhorn(scores, tau=tau, n_iters=n_iters))
    assert (plan >= 0).all()
    np.testing.assert_allclose(plan.sum(axis=1), np.ones(n), atol=1e-12)
    np.testing.assert_allclose(plan.sum(axis=0), np.ones(n), atol=atol)


def test_iterates_converge_monotonically_toward_double_stochasticity():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(10, 10))
    gaps = []
    for n_iters in (1, 2, 5, 20, 100):
        plan = np.asarray(sinkhorn(scores, tau=0.2, n_iters=n_iters))
        gaps.append(np.abs(plan.sum(axis=0) - 1.0).max())
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-8


def test_log_space_matches_the_literal_equation():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(7, 7))
    for tau in (1.0, 0.5):
        ours = np.asarray(sinkhorn(scores, tau=tau, n_iters=40))
        np.testing.assert_allclose(ours, naive_sinkhorn(scores, tau, 40), atol=1e-10)


def test_small_tau_agrees_with_the_hungarian_decode():
    """Section 3.3: tau -> 0 recovers a permutation matrix. On matrices with a
    clearly separated optimum that permutation is the one the decoder returns."""
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = 7
        # A random permutation plus a small perturbation, so the optimum wins by
        # a wide margin and the relaxation has an unambiguous target.
        perm = rng.permutation(n)
        scores = 8.0 * np.eye(n)[:, perm] + rng.normal(size=(n, n)) * 0.2
        plan = np.asarray(log_sinkhorn(scores, tau=0.02, n_iters=200))
        # P[i, j] = 1 when i == perm[j], so item perm[j] holds position j.
        assert np.array_equal(plan.argmax(axis=1), np.argsort(perm))
        assert np.array_equal(decode_ranking(scores), perm)


def test_small_tau_matches_hungarian_on_random_well_separated_matrices():
    rng = np.random.default_rng(5)
    agree = 0
    trials = 40
    for _ in range(trials):
        base = rng.normal(size=(6, 6))
        # Push the optimum away from the runners up by boosting its entries.
        order = decode_ranking(base)
        scores = base.copy()
        scores[order, np.arange(6)] += 6.0
        plan = np.asarray(log_sinkhorn(scores, tau=0.05, n_iters=300))
        agree += int(np.array_equal(plan.argmax(axis=1), np.argsort(order)))
    assert agree == trials


def test_loss_is_minimized_by_the_teacher_permutation():
    rng = np.random.default_rng(6)
    n = 6
    perm = rng.permutation(n)
    teacher = np.eye(n)[:, perm]
    aligned = 10.0 * teacher
    wrong = 10.0 * np.eye(n)[:, rng.permutation(n)]
    assert float(sinkhorn_loss(aligned, teacher, tau=0.5, n_iters=50)) < 1e-6
    assert float(sinkhorn_loss(wrong, teacher, tau=0.5, n_iters=50)) > 1.0
    assert float(sinkhorn_loss(np.zeros((n, n)), teacher, tau=0.5, n_iters=50)) == (
        pytest.approx(np.log(n))
    )


def test_loss_is_a_mean_of_per_item_cross_entropies():
    """Rows of a doubly stochastic plan are distributions, so equation (3)
    divided by N is bounded below by 0 and above by log N at the uniform plan."""
    rng = np.random.default_rng(8)
    n = 9
    scores = rng.normal(size=(n, n))
    teacher = np.eye(n)[:, rng.permutation(n)]
    loss = float(sinkhorn_loss(scores, teacher, tau=1.0, n_iters=100))
    log_plan = log_sinkhorn(scores, tau=1.0, n_iters=100)
    by_hand = -float(jnp.sum(jnp.asarray(teacher) * log_plan)) / n
    assert loss == pytest.approx(by_hand)
    assert loss > 0.0


def test_rejects_non_square_scores():
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((3, 5)))


def test_gradient_matches_central_differences():
    """d L / d M against central finite differences, entry by entry."""
    rng = np.random.default_rng(9)
    n = 5
    scores = jnp.asarray(rng.normal(size=(n, n)))
    teacher = jnp.asarray(np.eye(n)[:, rng.permutation(n)])

    def loss(m):
        return sinkhorn_loss(m, teacher, tau=0.7, n_iters=30)

    analytic = np.asarray(jax.grad(loss)(scores))
    eps = 1e-6
    numeric = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bump = jnp.zeros((n, n)).at[i, j].set(eps)
            numeric[i, j] = (float(loss(scores + bump)) - float(loss(scores - bump))) / (
                2 * eps
            )
    np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-5)


def test_gradient_is_finite_at_small_tau():
    """The log space form is the reason this does not blow up."""
    rng = np.random.default_rng(10)
    scores = jnp.asarray(rng.normal(size=(8, 8)) * 3.0)
    teacher = jnp.asarray(np.eye(8)[:, rng.permutation(8)])
    for tau in (0.5, 0.05, 0.01):
        g = jax.grad(lambda m: sinkhorn_loss(m, teacher, tau=tau, n_iters=50))(scores)
        assert np.isfinite(np.asarray(g)).all()


def test_gradient_points_toward_the_teacher():
    """One gradient step on M has to lower the loss and move M's decode toward
    the teacher's permutation."""
    rng = np.random.default_rng(12)
    n = 6
    perm = rng.permutation(n)
    teacher = np.eye(n)[:, perm]
    scores = jnp.asarray(rng.normal(size=(n, n)) * 0.1)
    def loss_fn(m):
        return sinkhorn_loss(m, teacher, tau=0.5, n_iters=30)

    grad_fn = jax.jit(jax.grad(loss_fn))
    before = float(loss_fn(scores))
    stepped = scores
    for _ in range(200):
        stepped = stepped - 0.5 * grad_fn(stepped)
    after = float(loss_fn(stepped))
    assert after < before
    assert np.array_equal(decode_ranking(np.asarray(stepped)), perm)
