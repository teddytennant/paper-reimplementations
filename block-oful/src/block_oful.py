"""Block-OFUL: truncated block features, OFUL block updates, confidence width.

Paper indices on a block U = (v_1, ..., v_{m+L}) are 1-based. Arrays are
0-based: U[j - 1] is v_j. Actions with time index <= 0 are treated as 0.
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ENUM_CAP = 4096


def vec_F(G):
    """Column-major vectorization, Fortran order.

    v^T G u = dot(kron(u, v), vec_F(G)).
    """
    G = jnp.asarray(G)
    return jnp.reshape(jnp.transpose(G), (-1,))


def impulse_responses(A, B, C, m):
    """G_h = C @ A^h @ B for h = 0, ..., m-1. Shape (m, p, p)."""
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    C = jnp.asarray(C)
    n = A.shape[0]
    Ah = jnp.eye(n, dtype=A.dtype)
    blocks = []
    for _ in range(int(m)):
        blocks.append(C @ Ah @ B)
        Ah = A @ Ah
    return jnp.stack(blocks, axis=0)


def theta_from_impulses(Gs):
    """theta_m = concat(vec_F(G_0), ..., vec_F(G_{m-1}))."""
    Gs = jnp.asarray(Gs)
    parts = [vec_F(Gs[h]) for h in range(Gs.shape[0])]
    return jnp.concatenate(parts, axis=0)


def theta_m(A, B, C, m):
    return theta_from_impulses(impulse_responses(A, B, C, m))


def phi_at(U, j, m):
    """phi_j in R^{m p^2} for 1-based index j.

    The s-th block, s = 1..m, is kron(v_{j-s}, v_j) if j - s >= 1, else 0.
    """
    U = jnp.asarray(U)
    p = U.shape[-1]
    v_j = U[j - 1]
    blocks = []
    zeros = jnp.zeros((p * p,), dtype=U.dtype)
    for s in range(1, int(m) + 1):
        lag = j - s
        if lag >= 1:
            blocks.append(jnp.kron(U[lag - 1], v_j))
        else:
            blocks.append(zeros)
    return jnp.concatenate(blocks, axis=0)


def suffix_features(U, m):
    """Features for paper indices j = m+1, ..., m+L. Shape (L, d)."""
    U = jnp.asarray(U)
    length = int(U.shape[0])
    m = int(m)
    if length <= m:
        raise ValueError("block length must exceed the burn-in m")
    return jnp.stack([phi_at(U, j, m) for j in range(m + 1, length + 1)], axis=0)


def truncated_action_mean(U, Gs, j, m):
    """m-truncated action-driven mean at 1-based index j.

    sum_{s=1}^{m} v_j^T G_{s-1} v_{j-s}, with v_t = 0 for t < 1.
    Gs has shape (m, p, p).
    """
    U = jnp.asarray(U)
    Gs = jnp.asarray(Gs)
    v_j = U[j - 1]
    acc = jnp.zeros((), dtype=U.dtype)
    for s in range(1, int(m) + 1):
        lag = j - s
        if lag >= 1:
            acc = acc + jnp.dot(v_j, Gs[s - 1] @ U[lag - 1])
    return acc


def confidence_norm(phi, V):
    """||phi||_{V^{-1}} = sqrt(phi^T V^{-1} phi)."""
    sol = jnp.linalg.solve(V, phi)
    return jnp.sqrt(jnp.maximum(jnp.dot(phi, sol), 0.0))


def block_optimistic_value(U, theta, V_inv, gamma, m):
    """Sum over the suffix of phi^T theta + gamma * ||phi||_{V^{-1}}."""
    phis = suffix_features(U, m)
    means = phis @ theta
    quad = jnp.einsum("li,ij,lj->l", phis, V_inv, phis)
    widths = jnp.sqrt(jnp.maximum(quad, 0.0))
    return jnp.sum(means + gamma * widths)


def initial_statistics(m, p, lam, dtype=None):
    """V_0 = lam I, S_0 = 0, theta_0 = 0. d = m p^2."""
    if dtype is None:
        dtype = jnp.float64
    d = int(m) * int(p) * int(p)
    V = jnp.asarray(lam, dtype=dtype) * jnp.eye(d, dtype=dtype)
    S = jnp.zeros((d,), dtype=dtype)
    theta = jnp.zeros((d,), dtype=dtype)
    return V, S, theta


def oful_update(V, S, phis, rewards):
    """V += sum phi phi^T, S += sum phi * reward, theta = V^{-1} S."""
    phis = jnp.atleast_2d(jnp.asarray(phis, dtype=V.dtype))
    rewards = jnp.reshape(jnp.asarray(rewards, dtype=V.dtype), (-1,))
    if phis.shape[0] != rewards.shape[0]:
        raise ValueError("phis and rewards must have the same leading length")
    V_new = V + phis.T @ phis
    S_new = S + phis.T @ rewards
    theta = jnp.linalg.solve(V_new, S_new)
    return V_new, S_new, theta


def observe_block(V, S, U, rewards, m):
    """Update from one block, using suffix indices j = m+1..m+L only.

    rewards has length m+L (full block, prefix dropped) or L (suffix already).
    """
    U = jnp.asarray(U)
    rewards = jnp.asarray(rewards, dtype=V.dtype)
    length = int(U.shape[0])
    m = int(m)
    if rewards.shape[0] == length:
        rewards = rewards[m:]
    elif rewards.shape[0] != length - m:
        raise ValueError("rewards must have length m+L or L")
    phis = suffix_features(U, m)
    return oful_update(V, S, phis, rewards)


def _as_blocks(action_set, length):
    action_set = np.asarray(action_set)
    k = int(action_set.shape[0])
    length = int(length)
    count = k ** length
    if count > ENUM_CAP:
        raise ValueError(
            f"exact enumeration cap is {ENUM_CAP}, got {k}^{length} = {count}"
        )
    axes = [np.arange(k) for _ in range(length)]
    grids = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    indices = grids.reshape(-1, length)
    blocks = action_set[indices]
    return jnp.asarray(blocks)


def exact_enum_oracle(action_set, m, L, theta, V, gamma):
    """Maximize the optimistic suffix value by enumerating action_set^{m+L}.

    Raises if the candidate count exceeds 4096.
    """
    length = int(m) + int(L)
    blocks = _as_blocks(action_set, length)
    theta = jnp.asarray(theta)
    V_inv = jnp.linalg.inv(jnp.asarray(V, dtype=theta.dtype))
    g = jnp.asarray(gamma, dtype=theta.dtype)
    m = int(m)

    def one(U):
        return block_optimistic_value(U, theta, V_inv, g, m)

    values = jax.vmap(one)(blocks)
    return blocks[jnp.argmax(values)]


def hypercube_actions(p):
    """All actions in {-1, +1}^p, shape (2^p, p)."""
    rows = list(itertools.product((-1.0, 1.0), repeat=int(p)))
    return jnp.asarray(rows, dtype=jnp.float64)


def local_search_oracle(theta, V, gamma, m, L, p, key, n_restarts=48):
    """Multi-start local search on {-1, +1}^p.

    Each start climbs by single-coordinate sign flips (best improvement)
    until a local maximum. The best start is kept. This is a heuristic,
    not an exact maximizer.
    """
    length = int(m) + int(L)
    p = int(p)
    m = int(m)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    V_inv = jnp.linalg.inv(jnp.asarray(V, dtype=jnp.float64))
    g = jnp.asarray(gamma, dtype=jnp.float64)
    n_flip = length * p

    @jax.jit
    def current_and_neighbors(U, theta_, V_inv_, g_):
        def flipped(idx):
            t = idx // p
            c = idx % p
            U2 = U.at[t, c].multiply(-1.0)
            return block_optimistic_value(U2, theta_, V_inv_, g_, m)

        cur = block_optimistic_value(U, theta_, V_inv_, g_, m)
        vals = jax.vmap(flipped)(jnp.arange(n_flip))
        return cur, vals

    def climb(U):
        # Finite hypercube and strict improvement imply termination.
        for _ in range(n_flip * 8):
            cur, vals = current_and_neighbors(U, theta, V_inv, g)
            bi = int(jnp.argmax(vals))
            if float(vals[bi]) <= float(cur) + 1e-12:
                return U, float(cur)
            t, c = divmod(bi, p)
            U = U.at[t, c].multiply(-1.0)
        cur, _ = current_and_neighbors(U, theta, V_inv, g)
        return U, float(cur)

    best_U = None
    best_val = -np.inf
    # A few canonical starts plus random restarts.
    seeds = [
        jnp.ones((length, p), dtype=jnp.float64),
        -jnp.ones((length, p), dtype=jnp.float64),
    ]
    checker = jnp.ones((length, p), dtype=jnp.float64)
    checker = checker.at[::2].set(-1.0)
    seeds.append(checker)
    for start in seeds:
        U, val = climb(start)
        if val > best_val:
            best_val = val
            best_U = U
    for _ in range(int(n_restarts)):
        key, sub = jax.random.split(key)
        bits = jax.random.bernoulli(sub, 0.5, shape=(length, p))
        start = jnp.where(bits, 1.0, -1.0)
        U, val = climb(start)
        if val > best_val:
            best_val = val
            best_U = U
    return best_U


def rollout_block(x, U, A, B, C, key=None, sigma_w=0.0, sigma_z=0.0):
    """Play U from state x. Reward uses x, then the linear step.

    Returns rewards of length m+L, the next state, and the updated key.
    """
    U = jnp.asarray(U)
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    C = jnp.asarray(C)
    x = jnp.asarray(x, dtype=A.dtype)
    rewards = []
    noisy = float(sigma_w) != 0.0 or float(sigma_z) != 0.0
    if noisy and key is None:
        key = jax.random.PRNGKey(0)
    for i in range(int(U.shape[0])):
        u = U[i]
        if noisy:
            key, k1, k2 = jax.random.split(key, 3)
            w = float(sigma_w) * jax.random.normal(k1, x.shape, dtype=x.dtype)
            z = float(sigma_z) * jax.random.normal(k2, (), dtype=x.dtype)
        else:
            w = jnp.zeros_like(x)
            z = jnp.zeros((), dtype=x.dtype)
        reward = jnp.dot(u, C @ x) + z
        x = A @ x + B @ u + w
        rewards.append(reward)
    return jnp.stack(rewards), x, key


def _gamma_at(gamma, tau):
    if callable(gamma):
        return gamma(int(tau))
    return gamma


def run_block_oful(
    A,
    B,
    C,
    action_set,
    m,
    L,
    n_blocks,
    lam,
    gamma,
    oracle="exact",
    key=None,
    sigma_w=0.0,
    sigma_z=0.0,
    n_restarts=48,
):
    """Closed-loop Block-OFUL.

    gamma is a constant or a callable gamma(tau), where tau is the number
    of blocks already folded into V (0 on the first decision).
    oracle is "exact", "local", or a callable returning a block U.
    State starts at x_1 = 0 and carries across blocks.
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    B = jnp.asarray(B, dtype=jnp.float64)
    C = jnp.asarray(C, dtype=jnp.float64)
    action_set = jnp.asarray(action_set, dtype=jnp.float64)
    p = int(action_set.shape[-1])
    n = int(A.shape[0])
    m = int(m)
    L = int(L)
    V, S, theta = initial_statistics(m, p, lam)
    x = jnp.zeros((n,), dtype=jnp.float64)
    if key is None:
        key = jax.random.PRNGKey(0)
    played = []
    rewards = []
    for tau in range(int(n_blocks)):
        g = _gamma_at(gamma, tau)
        key, sub = jax.random.split(key)
        if oracle == "exact":
            U = exact_enum_oracle(action_set, m, L, theta, V, g)
        elif oracle == "local":
            U = local_search_oracle(
                theta, V, g, m, L, p, sub, n_restarts=n_restarts
            )
        elif callable(oracle):
            U = oracle(action_set, m, L, theta, V, g, sub)
        else:
            raise ValueError(f"unknown oracle: {oracle}")
        U = jnp.asarray(U, dtype=jnp.float64)
        block_rewards, x, key = rollout_block(
            x, U, A, B, C, key=key, sigma_w=sigma_w, sigma_z=sigma_z
        )
        V, S, theta = observe_block(V, S, U, block_rewards, m)
        played.append(U)
        rewards.append(block_rewards)
    actions = jnp.concatenate(played, axis=0)
    reward_arr = jnp.concatenate(rewards, axis=0)
    return {
        "actions": actions,
        "rewards": reward_arr,
        "blocks": jnp.stack(played, axis=0),
        "theta": theta,
        "V": V,
        "S": S,
        "x": x,
        "cumulative_reward": jnp.sum(reward_arr),
    }


def oful_width(tau, d, L, L_phi, lambda_, delta, sigma_B, S_m, Delta_bar, T_unused):
    """gamma = alpha + beta from the Block-OFUL regret proof.

    T_unused is part of the signature and does not enter the formula.
    log is the natural logarithm.
    """
    del T_unused
    tau = jnp.asarray(tau, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    L = jnp.asarray(L, dtype=jnp.float64)
    L_phi = jnp.asarray(L_phi, dtype=jnp.float64)
    lambda_ = jnp.asarray(lambda_, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    sigma_B = jnp.asarray(sigma_B, dtype=jnp.float64)
    S_m = jnp.asarray(S_m, dtype=jnp.float64)
    Delta_bar = jnp.asarray(Delta_bar, dtype=jnp.float64)
    log_det = jnp.log(1.0 + tau * L * L_phi**2 / (d * lambda_))
    alpha = sigma_B * jnp.sqrt(d * log_det + 2.0 * jnp.log(1.0 / delta)) + jnp.sqrt(
        lambda_
    ) * S_m
    beta = Delta_bar * jnp.sqrt(2.0 * tau * L * d * log_det)
    return alpha + beta


def _lambda_max(Sigma_w):
    Sigma_w = jnp.atleast_2d(jnp.asarray(Sigma_w, dtype=jnp.float64))
    return jnp.linalg.eigvalsh(Sigma_w)[-1]


def _op_norm(M):
    return jnp.linalg.norm(jnp.asarray(M, dtype=jnp.float64), ord=2)


def sigma_B(sigma_z, Sigma_w, C, M_rho, rho):
    """Paper sigma_B, the square root of the stated variance.

    sigma_B^2 = sigma_z^2 + lambda_max(Sigma_w) * ||C||_2^2 * M_rho^2 / (1-rho)^2
    """
    var = sigma_z**2 + _lambda_max(Sigma_w) * _op_norm(C) ** 2 * M_rho**2 / (
        1.0 - rho
    ) ** 2
    return jnp.sqrt(var)


def Delta_bar(C, B, Sigma_w, M_rho, rho, m, T, delta):
    """Paper Delta_bar.

    ||C|| ||B|| M_rho / (1-rho) * rho^m
    + 2 rho^m * sqrt(2 lambda_max(Sigma_w) ||C||^2 M_rho^2 / (1-rho^2)
                     * log(2 * 5^p * T / delta))
    p is the action dimension, the row count of C. Norms are spectral.
    """
    C = jnp.asarray(C, dtype=jnp.float64)
    p = int(C.shape[0])
    C_op = _op_norm(C)
    B_op = _op_norm(B)
    lam = _lambda_max(Sigma_w)
    bias = C_op * B_op * M_rho / (1.0 - rho) * rho**m
    inner = (
        2.0
        * lam
        * C_op**2
        * M_rho**2
        / (1.0 - rho**2)
        * jnp.log(2.0 * (5.0**p) * T / delta)
    )
    noise = 2.0 * (rho**m) * jnp.sqrt(inner)
    return bias + noise
