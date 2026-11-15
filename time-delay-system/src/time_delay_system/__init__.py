"""Time-delay sequence model (arXiv:2610.12864).

Follows Algorithms 1, 2, and 6, not the conflicting main-text equations.
The diagonal state uses the parallel cumsum form. That form is for short T;
it is not a stability claim when the decay a is near 0 on long sequences.
The lag estimator is windowed over L past tokens, O(T L), not O(1).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "LagOutput",
    "TDSOutput",
    "adam_init",
    "adam_update",
    "apply",
    "cross_entropy",
    "decay_weights",
    "delay_correction",
    "diagonal_state_parallel",
    "diagonal_state_sequential",
    "fuzzy_modulate",
    "history_window",
    "init_params",
    "layer_norm",
    "positive_softplus",
    "run_delayed_copy_smoke",
    "semantic_lag",
]


class LagOutput(NamedTuple):
    x_bar: jnp.ndarray
    s_hat: jnp.ndarray
    alpha: jnp.ndarray
    decay: jnp.ndarray
    sim: jnp.ndarray
    s_tilde: jnp.ndarray
    history: jnp.ndarray


class TDSOutput(NamedTuple):
    last: jnp.ndarray
    mean: jnp.ndarray
    s: jnp.ndarray
    s_out: jnp.ndarray
    h: jnp.ndarray
    d: jnp.ndarray
    z: jnp.ndarray
    b: jnp.ndarray
    a: jnp.ndarray
    x_bar: jnp.ndarray
    s_hat: jnp.ndarray
    alpha: jnp.ndarray
    mu: jnp.ndarray
    f: jnp.ndarray
    g: jnp.ndarray
    logits: jnp.ndarray


def decay_weights(lam: jnp.ndarray, alpha: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """w_u = exp(-lam * u ** alpha) for lags u = 1..n_lags.

    alpha is the positive decay exponent, not an attention weight.
    """
    dtype = jnp.result_type(lam, alpha, jnp.float32)
    u = jnp.arange(1, n_lags + 1, dtype=dtype)
    return jnp.exp(-jnp.asarray(lam, dtype=dtype) * u ** jnp.asarray(alpha, dtype=dtype))


@jax.custom_vjp
def _l2_norm(x: jnp.ndarray) -> jnp.ndarray:
    """||x||_2. Subgradient is 0 at the origin so zero-padded history is safe."""
    return jnp.sqrt(jnp.sum(x * x, axis=-1))


def _l2_norm_fwd(x):
    n = jnp.sqrt(jnp.sum(x * x, axis=-1))
    return n, (x, n)


def _l2_norm_bwd(res, g):
    x, n = res
    # x / ||x||, with 0 at ||x|| = 0 (a valid subgradient; JAX's raw norm is NaN there).
    inv = jnp.where(n > 0, 1.0 / jnp.where(n > 0, n, 1.0), 0.0)
    return (x * inv[..., None] * g[..., None],)


_l2_norm.defvjp(_l2_norm_fwd, _l2_norm_bwd)


def _cosine(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Cosine similarity with eps in the denominator. Zero vectors score 0."""
    num = jnp.sum(a * b, axis=-1)
    den = _l2_norm(a) * _l2_norm(b)
    return num / (den + eps)


def history_window(x: jnp.ndarray, n_lags: int) -> jnp.ndarray:
    """Strictly past tokens, zero-padded at the start.

    x: (B, T, D) -> history (B, T, L, D) with lag axis u = 1..L.
    history[:, t, u - 1] is x[:, t - u] if t - u >= 0, else 0.
    No future token is placed in the window.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    batch, time, dim = x.shape
    pad = jnp.zeros((batch, n_lags, dim), dtype=x.dtype)
    padded = jnp.concatenate([pad, x], axis=1)
    # padded index of x_{t-u} is (t - u) + n_lags = t + (n_lags - u).
    starts = n_lags - jnp.arange(1, n_lags + 1)
    idx = starts[:, None] + jnp.arange(time)[None, :]
    gathered = padded[:, idx, :]
    return jnp.swapaxes(gathered, 1, 2)


def semantic_lag(
    x: jnp.ndarray,
    W_proj: jnp.ndarray,
    lam: jnp.ndarray,
    kappa: jnp.ndarray,
    alpha: jnp.ndarray,
    n_lags: int,
    eps: float = 1e-8,
) -> LagOutput:
    """Algorithm 1 semantic lag estimator.

    At position t the window is the previous n_lags tokens (strictly past).
    """
    history = history_window(x, n_lags)
    proj_x = jnp.einsum("pd,btd->btp", W_proj, x)
    proj_h = jnp.einsum("pd,btld->btlp", W_proj, history)
    sim = _cosine(proj_x[:, :, None, :], proj_h, eps=eps)
    decay = decay_weights(lam, alpha, n_lags).astype(sim.dtype)
    s_tilde = sim * decay
    attn = jax.nn.softmax(kappa * s_tilde, axis=-1)
    x_bar = jnp.sum(attn[..., None] * history, axis=2)
    s_hat = jnp.sum(attn * s_tilde, axis=-1)
    return LagOutput(
        x_bar=x_bar,
        s_hat=s_hat,
        alpha=attn,
        decay=decay,
        sim=sim,
        s_tilde=s_tilde,
        history=history,
    )


def fuzzy_modulate(
    z: jnp.ndarray,
    centers: jnp.ndarray,
    sigma: jnp.ndarray,
    f_r: jnp.ndarray,
    g_r: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Algorithm 2 fuzzy modulation.

    mu_tilde_r = exp(-||z - c_r||^2 / (2 sigma^2))
    mu_r = mu_tilde_r / (sum_j mu_tilde_j + 1e-8)
    f = sum_r mu_r f_r          (scalar, broadcast later)
    g = softplus(sum_r mu_r g_r)
    """
    diff = z[..., None, :] - centers
    dist2 = jnp.sum(diff * diff, axis=-1)
    mu_tilde = jnp.exp(-dist2 / (2.0 * sigma ** 2))
    mu = mu_tilde / (jnp.sum(mu_tilde, axis=-1, keepdims=True) + 1e-8)
    f = jnp.sum(mu * f_r, axis=-1)
    g = jax.nn.softplus(jnp.sum(mu * g_r, axis=-1))
    return mu, f, g


def delay_correction(
    s_hat: jnp.ndarray,
    x_bar: jnp.ndarray,
    gamma: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    """Algorithm 6: d = gamma * tanh(s_hat) * (C @ x_bar)."""
    cx = jnp.einsum("hd,...d->...h", C, x_bar)
    return gamma * jnp.tanh(s_hat)[..., None] * cx


def diagonal_state_parallel(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Closed form of h_t = a * h_{t-1} + b_t with h_0 = 0.

    h_t = a^{t} * cumsum_{i=1..t}(b_i / a^{i})
        = a^{⊙t} ⊙ cumsum(a^{⊙-i} ⊙ b_i)

    a: (H,) in (0, 1), constant over time.
    b: (B, T, H)
    Time index t = 1..T occupies axis 1.

    Short T only. Do not use this cumsum when a is near 0 and T is long:
    a^{⊙-i} overflows. The smoke and tests stay at T <= 16.
    """
    dtype = b.dtype
    a = jnp.asarray(a, dtype=dtype)
    time = b.shape[1]
    idx = jnp.arange(1, time + 1, dtype=dtype)
    log_a = jnp.log(a)
    a_pow = jnp.exp(idx[:, None] * log_a[None, :])
    a_inv = jnp.exp(-idx[:, None] * log_a[None, :])
    scaled = b * a_inv[None, :, :]
    return a_pow[None, :, :] * jnp.cumsum(scaled, axis=1)


def diagonal_state_sequential(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Reference recurrence h_t = a * h_{t-1} + b_t, h_0 = 0."""
    dtype = b.dtype
    a = jnp.asarray(a, dtype=dtype)

    def step(h, b_t):
        h = a * h + b_t
        return h, h

    h0 = jnp.zeros((b.shape[0], b.shape[-1]), dtype=dtype)
    _, hs = jax.lax.scan(step, h0, jnp.swapaxes(b, 0, 1))
    return jnp.swapaxes(hs, 0, 1)


def layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray | None = None,
    bias: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    y = (x - mean) / jnp.sqrt(var + eps)
    if scale is not None:
        y = y * scale
    if bias is not None:
        y = y + bias
    return y


def positive_softplus(raw: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(raw)


def _logit(p: float) -> jnp.ndarray:
    p_arr = jnp.asarray(p, dtype=jnp.float32)
    return jnp.log(p_arr) - jnp.log1p(-p_arr)


def _softplus_inv(y: float) -> jnp.ndarray:
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    return jnp.log(jnp.exp(y_arr) - 1.0)


def _glorot(key: jax.Array, out: int, inn: int) -> jnp.ndarray:
    limit = jnp.sqrt(jnp.asarray(6.0, dtype=jnp.float32) / (out + inn))
    return jax.random.uniform(key, (out, inn), minval=-limit, maxval=limit, dtype=jnp.float32)


def init_params(
    key: jax.Array,
    *,
    vocab: int = 32,
    d: int = 32,
    n_classes: int = 4,
    n_lags: int = 4,
    n_rules: int = 4,
    d_hidden: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Learned parameters. lam, kappa, alpha, sigma are softplus-constrained."""
    del n_lags  # window length is a static apply argument, not a tensor
    if d_hidden is None:
        d_hidden = d
    keys = jax.random.split(key, 6)
    params: dict[str, jnp.ndarray] = {
        "embed": jax.random.normal(keys[0], (vocab, d), dtype=jnp.float32) * 0.02,
        "W_proj": _glorot(keys[1], d, d),
        "lam_raw": jnp.asarray(0.0, dtype=jnp.float32),
        "kappa_raw": _softplus_inv(1.0),
        "alpha_raw": _softplus_inv(1.0),
        "gamma": jnp.asarray(1.0, dtype=jnp.float32),
        "C": _glorot(keys[2], d_hidden, d),
        "B": _glorot(keys[3], d_hidden, d),
        "a_raw": jnp.full((d_hidden,), _logit(0.9), dtype=jnp.float32),
        "centers": jax.random.normal(keys[4], (n_rules, d_hidden), dtype=jnp.float32) * 0.1,
        "sigma_raw": _softplus_inv(1.0),
        "f_r": jnp.ones((n_rules,), dtype=jnp.float32),
        "g_r": jnp.zeros((n_rules,), dtype=jnp.float32),
        "W_out": _glorot(keys[5], d, d_hidden),
        "ln_scale": jnp.ones((d,), dtype=jnp.float32),
        "ln_bias": jnp.zeros((d,), dtype=jnp.float32),
        "W_head": jnp.zeros((n_classes, d_hidden), dtype=jnp.float32),
        "b_head": jnp.zeros((n_classes,), dtype=jnp.float32),
    }
    return params


def apply(
    params: dict[str, jnp.ndarray],
    tokens: jnp.ndarray,
    n_lags: int,
) -> TDSOutput:
    """One time-delay block plus a linear head on the mean state.

    Returns last state s[:, -1] and the mean of s over time.
    s_out = LayerNorm(W_out @ s + x) is also returned.
    z = d / 2 (Algorithms 3 and 6), not (s_{k-1} + d) / 2.
    """
    x = params["embed"][tokens]
    lam = positive_softplus(params["lam_raw"])
    kappa = positive_softplus(params["kappa_raw"])
    alpha = positive_softplus(params["alpha_raw"])
    sigma = positive_softplus(params["sigma_raw"])

    lag = semantic_lag(x, params["W_proj"], lam, kappa, alpha, n_lags)
    d_corr = delay_correction(lag.s_hat, lag.x_bar, params["gamma"], params["C"])
    b = jnp.einsum("hd,btd->bth", params["B"], x)
    a = jax.nn.sigmoid(params["a_raw"])
    h = diagonal_state_parallel(a, b)

    z = d_corr / 2.0
    mu, f, g = fuzzy_modulate(z, params["centers"], sigma, params["f_r"], params["g_r"])
    s = f[..., None] * jax.nn.silu(h) + g[..., None] * d_corr

    proj = jnp.einsum("oh,bth->bto", params["W_out"], s)
    if "W_skip" in params:
        x_res = jnp.einsum("od,btd->bto", params["W_skip"], x)
    else:
        x_res = x
    s_out = layer_norm(proj + x_res, params["ln_scale"], params["ln_bias"])

    last = s[:, -1]
    mean = jnp.mean(s, axis=1)
    logits = jnp.einsum("bh,ch->bc", mean, params["W_head"]) + params["b_head"]
    return TDSOutput(
        last=last,
        mean=mean,
        s=s,
        s_out=s_out,
        h=h,
        d=d_corr,
        z=z,
        b=b,
        a=a,
        x_bar=lag.x_bar,
        s_hat=lag.s_hat,
        alpha=lag.alpha,
        mu=mu,
        f=f,
        g=g,
        logits=logits,
    )


def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits)
    idx = jnp.arange(labels.shape[0])
    return -jnp.mean(logp[idx, labels])


def adam_init(params: dict[str, jnp.ndarray]) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray], jnp.ndarray]:
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t = jnp.asarray(0, dtype=jnp.int32)
    return m, v, t


def adam_update(
    params: dict[str, jnp.ndarray],
    m: dict[str, jnp.ndarray],
    v: dict[str, jnp.ndarray],
    t: jnp.ndarray,
    grads: dict[str, jnp.ndarray],
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray], dict[str, jnp.ndarray], jnp.ndarray]:
    t = t + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * jnp.square(g), v, grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, m, v, t


def run_delayed_copy_smoke(
    steps: int = 40,
    seed: int = 0,
    vocab: int = 32,
    seq_len: int = 16,
    d: int = 32,
    n_lags: int = 4,
    n_rules: int = 4,
    n_classes: int = 4,
    batch: int = 64,
    lr: float = 1e-2,
    delay: int = 3,
) -> dict[str, Any]:
    """CPU Adam on a synthetic delayed-copy label. Not IMDB and not 99.1%.

    Label is the token `delay` steps before the last position, modulo n_classes.
    That token sits inside the lag window of later positions (delay < n_lags).
    Returns the loss before any update and the loss after `steps` updates.
    """
    if delay < 1 or delay >= seq_len:
        raise ValueError("delay must reference a strictly earlier token")
    key = jax.random.PRNGKey(seed)
    key, k_params, k_data = jax.random.split(key, 3)
    params = init_params(
        k_params,
        vocab=vocab,
        d=d,
        n_classes=n_classes,
        n_lags=n_lags,
        n_rules=n_rules,
        d_hidden=d,
    )
    tokens = jax.random.randint(k_data, (batch, seq_len), 0, vocab)
    labels = tokens[:, -1 - delay] % n_classes
    m, v, t = adam_init(params)

    def loss_fn(p, tokens, labels):
        out = apply(p, tokens, n_lags)
        return cross_entropy(out.logits, labels)

    @jax.jit
    def step(params, m, v, t, tokens, labels):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, labels)
        params, m, v, t = adam_update(params, m, v, t, grads, lr=lr)
        return params, m, v, t, loss

    loss0 = loss_fn(params, tokens, labels)
    last_pre = loss0
    for _ in range(steps):
        params, m, v, t, last_pre = step(params, m, v, t, tokens, labels)
    loss_final = loss_fn(params, tokens, labels)
    return {
        "step_0": float(loss0),
        "step_final": float(loss_final),
        "steps": steps,
        "pre_update_loss_at_last_step": float(last_pre),
        "finite": bool(jnp.isfinite(loss0) and jnp.isfinite(loss_final)),
    }
