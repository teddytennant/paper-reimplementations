"""R-contamination robust Bellman operator and a tiny encoder warm start.

Toy of Banerjee and Tsukamoto, arXiv:2609.23775. The operator is their
R-contamination backup. The encoder is one attention layer, not BERT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

N = 5
S = 25
A = 4
D = 16
# N, E, S, W
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _step(s: int, a: int) -> int:
    row, col = divmod(s, N)
    dr, dc = _DELTAS[a]
    nr, nc = row + dr, col + dc
    if not (0 <= nr < N and 0 <= nc < N):
        return s
    return nr * N + nc


def maze_kernel(goal: int = 24, slip: float = 0.1):
    """Nominal kernel P[s, a, s'] and reward r[s, a] in [0, 1].

    Intended move has probability 1-slip. The slip mass is uniform over the
    four actions. Entering the goal pays 1; the goal then absorbs with reward 0.
    """
    goal = int(goal)
    p = [[[0.0 for _ in range(S)] for _ in range(A)] for _ in range(S)]
    reward = [[0.0 for _ in range(A)] for _ in range(S)]
    for s in range(S):
        if s == goal:
            for a in range(A):
                p[s][a][s] = 1.0
            continue
        for a in range(A):
            intended = _step(s, a)
            for b in range(A):
                nxt = _step(s, b)
                prob = (1.0 - slip) if b == a else 0.0
                prob += slip / A
                p[s][a][nxt] += prob
            reward[s][a] = 1.0 if intended == goal else 0.0
    return jnp.asarray(p), jnp.asarray(reward)


def contamination_next(V, observed, R):
    """Paper's worst-case next value: R * min V + (1-R) * observed.

    `observed` is either V(s') (one sample) or E_{s'~P}[V] (known kernel).
    """
    return R * jnp.min(V) + (1.0 - R) * observed


def robust_backup(Q, r, P, gamma, R):
    """Expectation form of the R-contamination Bellman operator."""
    V = jnp.max(Q, axis=-1)
    expected = jnp.einsum("saj,j->sa", P, V)
    return r + gamma * contamination_next(V, expected, R)


def nominal_backup(Q, r, P, gamma):
    return robust_backup(Q, r, P, gamma, 0.0)


def bellman_residual(Q, r, P, gamma, R):
    return jnp.max(jnp.abs(Q - robust_backup(Q, r, P, gamma, R)))


def value_iteration(r, P, gamma, R, n: int = 40):
    def body(Q, _):
        return robust_backup(Q, r, P, gamma, R), None

    Q, _ = jax.lax.scan(body, jnp.zeros_like(r), None, length=n)
    return Q


def conformal_quantile(scores, delta):
    """ceil((1-delta)*(N+1))-th smallest of {scores} union {+inf}."""
    scores = jnp.asarray(scores, dtype=jnp.float32).reshape(-1)
    n = scores.shape[0]
    rank = jnp.ceil((1.0 - delta) * (n + 1)).astype(jnp.int32)
    pool = jnp.concatenate([scores, jnp.asarray([jnp.inf], dtype=scores.dtype)])
    ordered = jnp.sort(pool)
    idx = jnp.minimum(rank, pool.shape[0]) - 1
    return ordered[idx]


def stepwise_score(delta_q_t, delta_q_0, rho, k, gamma):
    """Step-wise nonconformity: max(0, ΔQ_t - ΔQ_0 * (1-ρ)^k / (1-γ))."""
    allowance = delta_q_0 * (1.0 - rho) ** k / (1.0 - gamma)
    return jnp.maximum(0.0, delta_q_t - allowance)


def init_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "W1": jax.random.normal(k1, (7, D)) * 0.05,
        "b1": jnp.zeros((D,)),
        "Wq": jax.random.normal(k2, (D, D)) * 0.05,
        "Wk": jax.random.normal(k3, (D, D)) * 0.05,
        "Wv": jax.random.normal(k4, (D, D)) * 0.05,
        "Wo": jnp.zeros((D, A)),
    }


def _features(goal, R):
    goal = jnp.asarray(goal, dtype=jnp.int32)
    gr = goal // N
    gc = goal % N
    idx = jnp.arange(S)
    rows = idx // N
    cols = idx % N
    return jnp.stack(
        [
            rows / 4.0,
            cols / 4.0,
            jnp.full((S,), gr / 4.0),
            jnp.full((S,), gc / 4.0),
            jnp.full((S,), R),
            jnp.abs(rows - gr) / 4.0,
            jnp.abs(cols - gc) / 4.0,
        ],
        axis=-1,
    )


def warm_start_q(goal, R, params):
    """One-layer encoder over the 25 cells, conditioned on the goal and R."""
    h = jax.nn.relu(_features(goal, R) @ params["W1"] + params["b1"])
    query = h @ params["Wq"]
    key = h @ params["Wk"]
    value = h @ params["Wv"]
    attn = jax.nn.softmax((query @ key.T) / jnp.sqrt(D), axis=-1)
    mixed = h + attn @ value
    return mixed @ params["Wo"]


def q_star(goal, R, gamma, slip=0.1, n=40):
    P, r = maze_kernel(int(goal), slip)
    return value_iteration(r, P, gamma, R, n=n)


def train_warm_start(goals, R, gamma, steps=120, lr=0.1, key=None, slip=0.1):
    """Plain gradient descent. No optax. Targets are robust Q* for each goal."""
    key = jax.random.PRNGKey(0) if key is None else key
    params = init_params(key)
    goals = [int(g) for g in goals]
    targets = jnp.stack([q_star(g, R, gamma, slip) for g in goals])
    goals_arr = jnp.asarray(goals, dtype=jnp.int32)

    def loss(params):
        preds = jax.vmap(lambda g: warm_start_q(g, R, params))(goals_arr)
        return jnp.mean((preds - targets) ** 2)

    grad_fn = jax.grad(loss)

    def step(params):
        grads = grad_fn(params)
        return jax.tree.map(lambda p, dp: p - lr * dp, params, grads)

    step = jax.jit(step)
    for _ in range(int(steps)):
        params = step(params)
    return params
