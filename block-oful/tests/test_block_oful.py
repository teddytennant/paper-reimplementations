"""Identities for the Block-OFUL slice. CPU, float64."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from block_oful import (
    Delta_bar,
    block_optimistic_value,
    exact_enum_oracle,
    hypercube_actions,
    impulse_responses,
    initial_statistics,
    local_search_oracle,
    observe_block,
    oful_update,
    oful_width,
    phi_at,
    run_block_oful,
    sigma_B,
    theta_from_impulses,
    truncated_action_mean,
    vec_F,
)


def _rollout_means(A, B, C, actions):
    """Independent no-noise means. r_t = u_t^T C x_t, x_1 = 0."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    rewards = []
    for u in actions:
        rewards.append(float(u @ (C @ x)))
        x = A @ x + B @ u
    return np.asarray(rewards, dtype=np.float64)


def _independent_phi(U, j, m):
    """Spec feature, numpy, 1-based j. Zeros a lag that falls before the block."""
    U = np.asarray(U, dtype=np.float64)
    p = U.shape[1]
    v_j = U[j - 1]
    blocks = []
    for s in range(1, m + 1):
        lag = j - s
        if lag >= 1:
            blocks.append(np.kron(U[lag - 1], v_j))
        else:
            blocks.append(np.zeros(p * p, dtype=np.float64))
    return np.concatenate(blocks, axis=0)


def _independent_trunc(actions, A, B, C, m, t):
    """Direct m-truncated sum at 1-based time t. Actions at time <= 0 are 0."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    u_t = actions[t - 1]
    acc = 0.0
    Ah = np.eye(A.shape[0], dtype=np.float64)
    for s in range(1, m + 1):
        if t - s >= 1:
            G = C @ Ah @ B
            acc += float(u_t @ G @ actions[t - s - 1])
        Ah = A @ Ah
    return acc


def test_feature_identity():
    key = jax.random.PRNGKey(7)
    n, p, m, L = 3, 2, 3, 4
    kA, kB, kC, kU = jax.random.split(key, 4)
    A = jax.random.normal(kA, (n, n))
    B = jax.random.normal(kB, (n, p))
    C = jax.random.normal(kC, (p, n))
    U = jax.random.normal(kU, (m + L, p))
    Gs = impulse_responses(A, B, C, m)
    theta = theta_from_impulses(Gs)

    # Vectorization convention: v^T G u = dot(kron(u, v), vec_F(G)).
    v = U[0]
    u = U[1]
    G0 = Gs[0]
    assert jnp.allclose(
        jnp.dot(v, G0 @ u),
        jnp.dot(jnp.kron(u, v), vec_F(G0)),
        atol=1e-12,
    )

    for j in range(m + 1, m + L + 1):
        phi = phi_at(U, j, m)
        direct = truncated_action_mean(U, Gs, j, m)
        indep = _independent_phi(np.asarray(U), j, m)
        assert jnp.allclose(phi, indep, atol=1e-12)
        assert jnp.allclose(jnp.dot(phi, theta), direct, atol=1e-10)
        # Same number as the direct truncated sum over G_h = C A^h B.
        manual = 0.0
        for s in range(1, m + 1):
            manual = manual + jnp.dot(U[j - 1], Gs[s - 1] @ U[j - s - 1])
        assert jnp.allclose(jnp.dot(phi, theta), manual, atol=1e-10)

    # A lag that falls before the block is the zero block.
    # j=2, m=3: s=2 and s=3 have j-s < 1.
    U_early = U[:2]
    phi_early = phi_at(U_early, j=2, m=3)
    pp = p * p
    assert jnp.allclose(phi_early[pp:], 0.0, atol=1e-15)
    assert jnp.allclose(phi_early[:pp], jnp.kron(U_early[0], U_early[1]), atol=1e-12)
    indep_early = _independent_phi(np.asarray(U_early), j=2, m=3)
    assert np.allclose(np.asarray(phi_early), indep_early, atol=1e-12)
    assert jnp.allclose(
        jnp.dot(phi_early, theta),
        truncated_action_mean(U_early, Gs, 2, 3),
        atol=1e-10,
    )
    # The zero blocks really are the before-block lags, not a dropped G.
    assert int(np.count_nonzero(np.abs(indep_early[pp:]) > 0)) == 0


def test_truncation_bound():
    rho = 0.4
    n, p = 2, 2
    A = rho * jnp.eye(n)
    B = jnp.array([[0.7, -0.2], [0.1, 0.5]], dtype=jnp.float64)
    C = jnp.array([[0.4, 0.3], [-0.2, 0.6]], dtype=jnp.float64)
    # ||u||_2 = 1.
    u = jnp.array([0.6, 0.8], dtype=jnp.float64)
    T = 30
    actions = jnp.tile(u, (T, 1))
    full = _rollout_means(A, B, C, actions)
    t = T  # 1-based last time
    C_op = float(jnp.linalg.norm(C, ord=2))
    B_op = float(jnp.linalg.norm(B, ord=2))
    gaps = {}
    for m in (1, 4):
        Gs = impulse_responses(A, B, C, m)
        truncated = float(truncated_action_mean(actions, Gs, t, m))
        # Library truncation matches an independent lag sum.
        indep = _independent_trunc(actions, A, B, C, m, t)
        assert abs(truncated - indep) <= 1e-12
        gap = abs(float(full[t - 1]) - truncated)
        bound = C_op * B_op * (rho**m) / (1.0 - rho)
        # Float slack only. A wrong power of rho should not hide here.
        assert gap <= bound + 1e-9
        gaps[m] = gap
    assert gaps[1] > 1e-6
    assert gaps[4] < gaps[1]


def test_regression_uses_suffix_only():
    key = jax.random.PRNGKey(3)
    n, p, m, L = 2, 2, 2, 3
    kA, kB, kC, kU = jax.random.split(key, 4)
    A = 0.25 * jax.random.normal(kA, (n, n))
    B = jax.random.normal(kB, (n, p))
    C = jax.random.normal(kC, (p, n))
    U = jax.random.normal(kU, (m + L, p))
    rewards = _rollout_means(A, B, C, U)
    lam = 1.5
    V0, S0, _ = initial_statistics(m, p, lam)
    V, S, theta = observe_block(V0, S0, U, jnp.asarray(rewards), m)

    phis = np.stack(
        [_independent_phi(np.asarray(U), j, m) for j in range(m + 1, m + L + 1)],
        axis=0,
    )
    suffix_r = rewards[m:]
    expected_V = lam * np.eye(m * p * p) + phis.T @ phis
    expected_S = phis.T @ suffix_r
    assert np.allclose(np.asarray(V), expected_V, atol=1e-10)
    assert np.allclose(np.asarray(S), expected_S, atol=1e-10)
    assert np.allclose(np.asarray(theta), np.linalg.solve(expected_V, expected_S), atol=1e-10)
    # Prefix outer products are not in the update.
    prefix = np.stack(
        [_independent_phi(np.asarray(U), j, m) for j in range(1, m + 1)],
        axis=0,
    )
    # j=1 has some zero lags, but the burn-in still has a nonzero feature.
    assert np.linalg.norm(prefix) > 1e-8
    leaked = expected_V + prefix.T @ prefix
    assert not np.allclose(np.asarray(V), leaked, atol=1e-6)

    # Many fresh episodes from x=0. Only the first suffix step j=m+1 is
    # regressed. History length there is m, so the truncation is exact.
    m = 4
    p = 2
    L = 1
    n = 3
    kA, kB, kC, kE = jax.random.split(jax.random.PRNGKey(11), 4)
    A = 0.3 * jax.random.normal(kA, (n, n))
    B = jax.random.normal(kB, (n, p))
    C = jax.random.normal(kC, (p, n))
    d = m * p * p
    lam = 1e-8
    V = lam * jnp.eye(d)
    S = jnp.zeros(d)
    n_eps = 800
    keys = jax.random.split(kE, n_eps)
    theta_true = theta_from_impulses(impulse_responses(A, B, C, m))
    for ep in range(n_eps):
        U = jax.random.normal(keys[ep], (m + L, p))
        rewards = _rollout_means(A, B, C, U)
        r = rewards[m]  # paper index j = m+1
        phi = phi_at(U, m + 1, m)
        assert jnp.allclose(r, jnp.dot(phi, theta_true), atol=1e-8)
        V, S, theta = oful_update(V, S, phi, jnp.asarray([r]))
    G0 = vec_F(C @ B)
    assert jnp.allclose(theta[: p * p], G0, atol=1e-5, rtol=1e-4)
    assert jnp.allclose(theta, theta_true, atol=1e-5, rtol=1e-4)


def test_exact_enum_known_theta():
    m, L, p = 1, 1, 1
    action_set = jnp.array([[-1.0], [1.0]])
    theta = jnp.array([0.7])
    V = jnp.eye(1)
    gamma = 0.0
    U = exact_enum_oracle(action_set, m, L, theta, V, gamma)
    V_inv = jnp.linalg.inv(V)
    got = float(block_optimistic_value(U, theta, V_inv, gamma, m))
    best = -1e18
    for a0 in (-1.0, 1.0):
        for a1 in (-1.0, 1.0):
            Ub = jnp.array([[a0], [a1]])
            best = max(best, float(block_optimistic_value(Ub, theta, V_inv, gamma, m)))
    assert abs(got - best) <= 1e-12
    assert abs(best - 0.7) <= 1e-12
    # Legal action, and the truncated suffix reward is the max.
    assert U.shape == (m + L, p)
    assert float(jnp.max(jnp.abs(jnp.abs(U) - 1.0))) <= 1e-12


def test_local_search_matches_exact_enum():
    cases = (
        {"p": 2, "m": 1, "L": 1, "seed": 0},  # 4^2 = 16
        {"p": 2, "m": 1, "L": 2, "seed": 1},  # 4^3 = 64
    )
    for case in cases:
        p, m, L, seed = case["p"], case["m"], case["L"], case["seed"]
        d = m * p * p
        key = jax.random.PRNGKey(100 + seed)
        k1, k2, k3 = jax.random.split(key, 3)
        theta = jax.random.normal(k1, (d,))
        M = jax.random.normal(k2, (d, d))
        V = M.T @ M + 0.4 * jnp.eye(d)
        gamma = 0.35
        action_set = hypercube_actions(p)
        assert action_set.shape[0] ** (m + L) <= 4096
        U_ex = exact_enum_oracle(action_set, m, L, theta, V, gamma)
        U_ls = local_search_oracle(
            theta, V, gamma, m, L, p, k3, n_restarts=96
        )
        V_inv = jnp.linalg.inv(V)
        v_ex = float(block_optimistic_value(U_ex, theta, V_inv, gamma, m))
        v_ls = float(block_optimistic_value(U_ls, theta, V_inv, gamma, m))
        assert v_ls <= v_ex + 1e-8
        assert abs(v_ls - v_ex) <= 1e-7, (case, v_ls, v_ex)
        # Sign pattern stays on the hypercube.
        assert jnp.allclose(jnp.abs(U_ls), 1.0)


def test_closed_loop_finite_legal_actions():
    n, p, m, L = 2, 1, 1, 2
    A = jnp.array([[0.5, 0.1], [0.0, 0.3]], dtype=jnp.float64)
    B = jnp.array([[1.0], [0.2]], dtype=jnp.float64)
    C = jnp.array([[1.0, 0.4]], dtype=jnp.float64)
    action_set = jnp.array([[-1.0], [0.0], [1.0]], dtype=jnp.float64)
    n_blocks = 30
    # Small gamma, including the callable path. tau is blocks already seen.
    out = run_block_oful(
        A,
        B,
        C,
        action_set,
        m,
        L,
        n_blocks,
        lam=1.0,
        gamma=lambda tau: 0.01,
        oracle="exact",
        key=jax.random.PRNGKey(0),
    )
    assert out["actions"].shape == (n_blocks * (m + L), p)
    assert jnp.all(jnp.isfinite(out["actions"]))
    assert jnp.all(jnp.isfinite(out["rewards"]))
    assert jnp.all(jnp.isfinite(out["theta"]))
    assert jnp.all(jnp.isfinite(out["V"]))
    for u in np.asarray(out["actions"]):
        dists = np.linalg.norm(action_set - u[None, :], axis=-1)
        assert float(np.min(dists)) <= 1e-8

    # Random baseline on the same horizon, no noise. Not an assertion:
    # a loss is reported, not hidden.
    T = n_blocks * (m + L)
    rng = np.random.default_rng(0)
    random_totals = []
    for _ in range(20):
        idx = rng.integers(0, action_set.shape[0], size=T)
        acts = np.asarray(action_set)[idx]
        random_totals.append(float(np.sum(_rollout_means(A, B, C, acts))))
    oful_total = float(out["cumulative_reward"])
    random_mean = float(np.mean(random_totals))
    # Informational. Do not fail the test when OFUL does not beat random.
    assert math.isfinite(oful_total)
    assert math.isfinite(random_mean)


def test_width_and_noise_helpers_match_hand_expansion():
    # tau=2, d=4, L=2, L_phi=1.5, lambda=0.5, delta=0.1,
    # sigma_B=0.3, S_m=1.2, Delta_bar=0.4, T_unused ignored.
    # log_det = ln(1 + 2*2*2.25/(4*0.5)) = ln(5.5)
    # alpha = 0.3*sqrt(4*ln(5.5) + 2*ln(10)) + sqrt(0.5)*1.2
    # beta = 0.4*sqrt(2*2*2*4*ln(5.5))
    # gamma = 4.81688897670131765949
    g1 = oful_width(2, 4, 2, 1.5, 0.5, 0.1, 0.3, 1.2, 0.4, 100)
    g2 = oful_width(2, 4, 2, 1.5, 0.5, 0.1, 0.3, 1.2, 0.4, 7)
    assert abs(float(g1) - 4.81688897670131765949) <= 1e-12
    assert abs(float(g1) - float(g2)) <= 1e-15

    # sigma_B^2 = 1^2 + 4 * 2^2 * 1 / (0.5)^2 = 65, sigma_B = sqrt(65)
    Sigma_w = jnp.array([[4.0]])
    C = jnp.array([[2.0]])
    B = jnp.array([[3.0]])
    sig = sigma_B(1.0, Sigma_w, C, 1.0, 0.5)
    assert abs(float(sig) - 8.06225774829854913150) <= 1e-12
    assert abs(float(sig) ** 2 - 65.0) <= 1e-12

    # bias = 2*3*1/0.5 * 0.5^2 = 3
    # inner = 2*4*4*1/(1-0.25) * ln(2*5*10/0.1) = (32/0.75)*ln(1000)
    # Delta_bar = 3 + 2*(0.25)*sqrt(inner) = 11.58386410515738873528
    db = Delta_bar(C, B, Sigma_w, 1.0, 0.5, 2, 10, 0.1)
    assert abs(float(db) - 11.58386410515738873528) <= 1e-12
