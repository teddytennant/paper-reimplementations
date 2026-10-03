"""Path divergence and a tiny path-regularized score fine-tune.

The paper's Girsanov weight on a likelihood path is g(t)^2 (equation 2). The
task asks for the normalized weight

    c(t) = g_n(t)^2 / beta(t)
    A = 0.5 * int_0^1 c(t) ||s_FT(state_t, t, xi) - s_PT(state_t, t, xi)||^2 dt

and L = L_NLL + lambda_path * A. For n = 1, c(t) = 1, so this is not the same
numeric weight as 0.5 * int g(t)^2 ||.||^2. The integral above is what is
implemented.

Misspecification used by the fine-tune (1D, design-dependent bias):

    simulator:    y | theta, xi ~ N(theta, 1)
    observations: y | theta, xi ~ N(theta + b(xi), 1), b(xi) = 0.5 * xi

s_PT is the exact VP score of the simulator. s_FT is a small MLP. L_NLL is the
denoising score-matching loss against the observation score, a tractable NLL
surrogate. The paper's reward-through-sampler adjoint is not reproduced.

The diffused coordinate in A is the observation, conditioned on (theta, xi).
The task writes that coordinate as theta_t; here it is the response path, which
is the object in equation (2).
"""

import jax
import jax.numpy as jnp

from hbds.diffusion import diffusion_weight, signal_scale

HIDDEN = 16
BIAS_SCALE = 0.5


def _trapezoid(y, x):
    dx = jnp.diff(x)
    mid = 0.5 * (y[:-1] + y[1:])
    return jnp.sum(dx * mid)


def path_divergence(s_ft, s_pt, states, times, cond, n=1):
    """Quadrature of A along one provided trajectory.

    states: (T, D), times: (T,) increasing. Score callables have signature
    s(state, t, cond) -> (D,). A is 0 when the two scores agree on the path
    and positive when the integrand is not identically zero.
    """
    times = jnp.asarray(times, dtype=jnp.float64)
    states = jnp.asarray(states, dtype=jnp.float64)
    sq = []
    for i in range(int(times.shape[0])):
        diff = jnp.asarray(s_ft(states[i], times[i], cond), dtype=jnp.float64)
        diff = diff - jnp.asarray(s_pt(states[i], times[i], cond), dtype=jnp.float64)
        sq.append(jnp.sum(diff ** 2))
    sq = jnp.stack(sq)
    c = diffusion_weight(times, n)
    return _trapezoid(0.5 * c * sq, times)


def path_divergence_from_values(s_ft, s_pt, times, n=1):
    """Same integral when scores are already evaluated, shapes (T,) or (T, D)."""
    diff = jnp.asarray(s_ft, dtype=jnp.float64) - jnp.asarray(s_pt, dtype=jnp.float64)
    if diff.ndim == 1:
        sq = diff ** 2
    else:
        sq = jnp.sum(diff ** 2, axis=-1)
    c = diffusion_weight(jnp.asarray(times, dtype=jnp.float64), n)
    return _trapezoid(0.5 * c * sq, jnp.asarray(times, dtype=jnp.float64))


def dsm_nll(s_ft, s_target):
    """Mean squared error between a score and the denoising target."""
    return jnp.mean((jnp.asarray(s_ft) - jnp.asarray(s_target)) ** 2)


def path_regularized_loss(nll, path_a, lambda_path):
    """L = L_NLL + lambda_path * A."""
    return nll + lambda_path * path_a


def observation_bias(xi):
    return BIAS_SCALE * xi


def observation_marginal_score(y_t, t, mean):
    """Score of VP-noised N(mean, 1). Marginal variance is 1 for this pair."""
    a = signal_scale(t)
    return -(y_t - a * mean)


def simulator_score(y_t, t, theta, xi):
    """Pretrained score: simulator mean theta, design ignored."""
    del xi
    return observation_marginal_score(y_t, t, theta)


def observation_score(y_t, t, theta, xi):
    """Score of the misspecified observation model, mean theta + 0.5 xi."""
    return observation_marginal_score(y_t, t, theta + observation_bias(xi))


def init_score_mlp(key, in_dim=4, hidden=HIDDEN, out_dim=1):
    """A few hundred parameters. Last layer starts at zero so the initial score is 0."""
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (in_dim, hidden), dtype=jnp.float64) * jnp.sqrt(2.0 / in_dim)
    b1 = jnp.zeros((hidden,), dtype=jnp.float64)
    w2 = jax.random.normal(k2, (hidden, hidden), dtype=jnp.float64) * jnp.sqrt(2.0 / hidden)
    b2 = jnp.zeros((hidden,), dtype=jnp.float64)
    w3 = jnp.zeros((hidden, out_dim), dtype=jnp.float64)
    b3 = jnp.zeros((out_dim,), dtype=jnp.float64)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def mlp_param_count(params=None):
    if params is None:
        params = init_score_mlp(jax.random.PRNGKey(0))
    return int(sum(int(v.size) for v in jax.tree_util.tree_leaves(params)))


def mlp_score(params, y, t, theta, xi):
    """Score MLP. Inputs are scalars or matching batches. Returns the score."""
    x = jnp.stack(
        [
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(t, dtype=jnp.float64),
            jnp.asarray(theta, dtype=jnp.float64),
            jnp.asarray(xi, dtype=jnp.float64),
        ],
        axis=-1,
    )
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    out = h @ params["w3"] + params["b3"]
    return out[..., 0]


def sample_observation_batch(key, batch):
    """Noised observations from the true (biased) observation model."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    theta = jax.random.normal(k1, (batch,), dtype=jnp.float64)
    xi = jax.random.uniform(k2, (batch,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    t = jax.random.uniform(k3, (batch,), minval=1e-3, maxval=1.0, dtype=jnp.float64)
    y0 = theta + observation_bias(xi) + jax.random.normal(k4, (batch,), dtype=jnp.float64)
    a = signal_scale(t)
    sig = jnp.sqrt(jnp.maximum(1.0 - a ** 2, 0.0))
    y_t = a * y0 + sig * jax.random.normal(k5, (batch,), dtype=jnp.float64)
    return y_t, t, theta, xi


def batch_objective(s_ft_vals, s_target_vals, s_pt_vals, times, lambda_path, n=1):
    """L, L_NLL, A on already evaluated scores.

    If times are uniform on (0, 1), the mean of 0.5 * c(t) * ||diff||^2 estimates
    the path integral. The quadrature form is path_divergence.
    """
    nll = dsm_nll(s_ft_vals, s_target_vals)
    c = diffusion_weight(times, n)
    diff = jnp.asarray(s_ft_vals) - jnp.asarray(s_pt_vals)
    path_a = jnp.mean(0.5 * c * diff ** 2)
    return path_regularized_loss(nll, path_a, lambda_path), nll, path_a


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": jnp.array(0, dtype=jnp.int32)}


def adam_step(params, grads, state, lr):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda mm, g: 0.9 * mm + 0.1 * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: 0.999 * vv + 0.001 * (g ** 2), state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - 0.9 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - 0.999 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + 1e-8),
        params,
        mhat,
        vhat,
    )
    return new_params, {"m": m, "v": v, "t": t}


def train_step(params, opt_state, y_t, t, theta, xi, lambda_path, lr):
    """One Adam step on L = L_NLL + lambda_path * A, A the Monte Carlo path term."""

    def loss(p):
        s_ft = mlp_score(p, y_t, t, theta, xi)
        s_tgt = observation_score(y_t, t, theta, xi)
        s_pt = simulator_score(y_t, t, theta, xi)
        total, nll, path_a = batch_objective(s_ft, s_tgt, s_pt, t, lambda_path, n=1)
        return total, (nll, path_a)

    (total, aux), grads = jax.value_and_grad(loss, has_aux=True)(params)
    params, opt_state = adam_step(params, grads, opt_state, lr)
    return params, opt_state, total, aux


def run_finetune(steps=40, seed=0, lambda_path=0.1, lr=1e-2, batch=32):
    """CPU fine-tune of the score MLP. Returns params and the loss trace."""
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    params = init_score_mlp(k_init)
    opt = adam_init(params)

    def body(carry, _):
        params, opt, key = carry
        key, kb = jax.random.split(key)
        y_t, t, theta, xi = sample_observation_batch(kb, batch)
        params, opt, total, _aux = train_step(
            params, opt, y_t, t, theta, xi, lambda_path, lr
        )
        return (params, opt, key), total

    (params, _opt, _key), losses = jax.lax.scan(
        body, (params, opt, key), jnp.arange(steps)
    )
    return {"params": params, "losses": losses, "n_params": mlp_param_count(params)}
