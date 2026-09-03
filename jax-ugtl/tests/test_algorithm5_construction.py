"""Algorithm 5 invariants: no revisits, the score constraint of Eq. (22), the
Eq. (24) sampling law, and the effect of ensemble uncertainty on the temperature.
"""

import jax
import numpy as np
import pytest

from ugtl.trajectory import (
    construct_trajectories,
    cosine_alignment,
    score_feasible_neighbourhood,
    temperature,
    transition_probs,
)

HORIZON, N_TRAJ, K, XI = 12, 64, 10, 0.05


@pytest.fixture(scope="module")
def built(scaled_task, surrogate):
    X, y = scaled_task
    traj = construct_trajectories(
        jax.random.PRNGKey(2), X, y, surrogate,
        horizon=HORIZON, n_traj=N_TRAJ, K=K, xi=XI, rho=1.0,
    )
    return X, y, traj


def test_a_trajectory_never_revisits_a_design(built):
    _, _, traj = built
    for row in traj.indices:
        assert len(set(row.tolist())) == HORIZON


def test_every_step_is_score_feasible_or_used_the_documented_fallback(built):
    """Eq. (22) requires y_j >= b_k - xi; line 8 is the only way out of it."""
    _, y, traj = built
    for u in range(len(traj)):
        best = traj.scores[u, 0]
        for k in range(HORIZON - 1):
            nxt = traj.scores[u, k + 1]
            if not traj.fallback[u, k]:
                assert nxt >= best - XI - 1e-6
            best = max(best, nxt)


def test_retention_keeps_the_top_rho_by_endpoint_improvement(scaled_task, surrogate):
    """Lines 15 and 16: rank by Delta_y = y_H - y_1, keep ceil(rho N_traj)."""
    X, y = scaled_task
    kept = construct_trajectories(
        jax.random.PRNGKey(3), X, y, surrogate, horizon=HORIZON, n_traj=40, K=K, rho=0.25
    )
    assert len(kept) == 10
    assert np.all(np.diff(kept.improvement) <= 1e-6)
    everything = construct_trajectories(
        jax.random.PRNGKey(3), X, y, surrogate, horizon=HORIZON, n_traj=40, K=K, rho=1.0
    )
    assert kept.improvement.min() >= np.sort(everything.improvement)[-10] - 1e-6


def test_the_neighbourhood_is_the_k_nearest_feasible_designs(scaled_task):
    X, y = scaled_task
    rng = np.random.default_rng(0)
    cur = rng.integers(0, X.shape[0], size=5)
    dist2 = ((X[cur][:, None, :] - X[None, :, :]) ** 2).sum(-1)
    unvisited = np.ones((5, X.shape[0]), dtype=bool)
    unvisited[np.arange(5), cur] = False
    best = y[cur]
    picked, valid, fallback = score_feasible_neighbourhood(dist2, unvisited, y, best, XI, K)
    assert not fallback.any()
    for u in range(5):
        members = picked[u][valid[u]]
        assert (y[members] >= best[u] - XI).all()
        feasible = np.flatnonzero(unvisited[u] & (y >= best[u] - XI))
        expected = feasible[np.argsort(dist2[u, feasible], kind="stable")[:K]]
        assert set(members.tolist()) == set(expected.tolist())


def test_the_fallback_takes_the_highest_scoring_unvisited_designs(scaled_task):
    """No unvisited design clears the score constraint, so line 8 fires."""
    X, y = scaled_task
    n = X.shape[0]
    dist2 = ((X[:1] - X) ** 2).sum(-1)[None, :]
    unvisited = np.ones((1, n), dtype=bool)
    best = np.array([y.max() + 1.0], dtype=np.float32)  # nothing can be feasible
    picked, valid, fallback = score_feasible_neighbourhood(dist2, unvisited, y, best, XI, K)
    assert fallback[0]
    assert valid[0].all()
    assert set(picked[0].tolist()) == set(np.argsort(-y, kind="stable")[:K].tolist())


def test_sampling_frequencies_match_the_eq24_softmax():
    """Draw many times from Eq. (24) and compare with the closed form."""
    rng = np.random.default_rng(0)
    d, K = 3, 8
    grad = rng.normal(size=d)
    x_k = rng.normal(size=d)
    neighbours = rng.normal(size=(K, d))
    a = cosine_alignment(grad, x_k, neighbours)
    kappa = temperature(0.3, 0.4, 0.5)
    p = transition_probs(a, kappa)

    n_draws = 200_000
    cdf = np.cumsum(p)
    draws = np.searchsorted(cdf, rng.random(n_draws))
    freq = np.bincount(draws, minlength=K) / n_draws
    assert np.abs(freq - p).max() < 4.0 / np.sqrt(n_draws)


def test_the_alignment_is_the_stabilized_cosine():
    rng = np.random.default_rng(1)
    grad = rng.normal(size=4)
    x_k = rng.normal(size=4)
    neighbours = x_k + rng.normal(size=(6, 4))
    a = cosine_alignment(grad, x_k, neighbours)
    for j in range(6):
        delta = neighbours[j] - x_k
        expected = np.dot(grad, delta) / (np.linalg.norm(grad) * np.linalg.norm(delta) + 1e-8)
        assert a[j] == pytest.approx(expected, rel=1e-6)
    assert np.abs(a).max() <= 1.0 + 1e-9

    # A repeated design has zero displacement, which c_cos keeps finite.
    assert np.isfinite(cosine_alignment(grad, x_k, x_k[None, :])).all()

    # Perfect alignment with the gradient gives exactly 1 up to c_cos.
    assert cosine_alignment(grad, x_k, (x_k + grad)[None, :])[0] == pytest.approx(1.0, abs=1e-6)


def test_low_uncertainty_gives_a_lower_temperature_and_a_sharper_distribution():
    """kappa_k = 1 + sigma_phi / (sigma_bar + c_temp), so entropy rises with sigma."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=12) * 0.5
    sigma_bar, c_temp = 0.2, 0.5
    kappas, entropies = [], []
    for sigma in np.linspace(0.0, 3.0, 12):
        kappa = temperature(sigma, sigma_bar, c_temp)
        p = transition_probs(a, kappa)
        kappas.append(kappa)
        entropies.append(float(-(p * np.log(p)).sum()))
    assert kappas == sorted(kappas)
    assert kappas[0] == pytest.approx(1.0)
    assert np.all(np.diff(entropies) > 0)
    assert entropies[-1] < np.log(12)


def test_probabilities_ignore_slots_outside_the_neighbourhood():
    a = np.array([2.0, 1.0, 0.0, -5.0])
    valid = np.array([True, True, False, False])
    p = transition_probs(a, 1.0, valid)
    assert p[2] == 0.0 and p[3] == 0.0
    assert p.sum() == pytest.approx(1.0)
    assert p[0] / p[1] == pytest.approx(np.e, rel=1e-6)


def test_trajectories_start_in_the_bottom_percentile(built):
    _, y, traj = built
    cutoff = np.percentile(y, 20.0)
    assert (traj.scores[:, 0] <= cutoff + 1e-6).all()


def test_construction_climbs_and_stays_local(built):
    """The point of Eq. (22) plus the gradient bias: paths improve, and they do
    it with short steps rather than jumps across the design space."""
    X, _, traj = built
    assert traj.improvement.mean() > 0.0
    steps = np.linalg.norm(np.diff(traj.designs, axis=1), axis=-1)
    spread = np.linalg.norm(X.max(axis=0) - X.min(axis=0))
    assert steps.mean() < 0.1 * spread


def test_the_device_neighbourhood_agrees_with_the_reference(scaled_task):
    """``construct_trajectories`` runs Eq. (22) on device; it has to pick the same
    neighborhood as the numpy reference.

    The comparison is by distance, not by index. Two designs sitting the same
    distance from x_k are both correct answers for the last slot of S_k, and the
    two paths break that tie with different code. What is not allowed is picking
    a design that is measurably further out, which is how the TF32 matmul this
    used to run showed up: on an H200 it put the 11th nearest design in the
    neighborhood ahead of the 10th, 0.019 closer.
    """
    import jax.numpy as jnp

    from ugtl.trajectory import _nearest_feasible

    X, y = scaled_task
    rng = np.random.default_rng(5)
    m, n = 6, X.shape[0]
    cur = rng.integers(0, n, size=m)
    visited = rng.random((m, n)) < 0.3
    visited[np.arange(m), cur] = True
    best = y[cur]

    picked, valid, has_feasible = _nearest_feasible(
        jnp.asarray(X), jnp.asarray(y), jnp.asarray(cur), jnp.asarray(visited), jnp.asarray(best), K, XI
    )
    picked, valid = np.asarray(picked), np.asarray(valid)
    dist2 = ((X[cur][:, None, :] - X[None, :, :]) ** 2).sum(-1)
    ref_picked, ref_valid, ref_fallback = score_feasible_neighbourhood(dist2, ~visited, y, best, XI, K)

    # Judge both selections against float64 distances so the tolerance below is
    # about the tie, not about how either path rounded.
    X64 = X.astype(np.float64)
    exact = ((X64[cur][:, None, :] - X64[None, :, :]) ** 2).sum(-1)

    assert np.array_equal(np.asarray(has_feasible), ~ref_fallback)
    for u in range(m):
        assert np.array_equal(valid[u], ref_valid[u])
        got = picked[u][valid[u]]
        assert len(set(got.tolist())) == len(got)
        assert (~visited[u][got]).all()
        assert (y[got] >= best[u] - XI).all()
        # Same K distances as the true K nearest, so any index that differs from
        # the reference is a tie and not a worse pick.
        want = ref_picked[u][ref_valid[u]]
        assert np.allclose(np.sort(exact[u, got]), np.sort(exact[u, want]), rtol=0, atol=1e-6)
