"""Trajectory modeling: the conditional DDPM p_theta(X_tau | y_end).

The forward process is

    q(X^(t) | X^(0)) = N(sqrt(abar_t) X^(0), (1 - abar_t) I),

the training loss is the plain epsilon-prediction objective with the terminal
score dropped with probability p_drop in favour of a null-conditioning
embedding, and sampling uses classifier-free guidance (Eq. 25), the reverse step
of Eq. (26), and the prefix context conditioning of Eq. (27).

The denoiser is an MLP over the flattened design sequence rather than the
temporal U-Net of Section 5, which is the one architectural departure from the
paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax

from ugtl.nn import apply_mlp, init_mlp

__all__ = [
    "Schedule",
    "cosine_schedule",
    "q_sample",
    "DiffusionModel",
    "init_eps_model",
    "apply_eps_model",
    "guided_eps",
    "reverse_step",
    "clamp_prediction",
    "train_diffusion",
    "sample_trajectories",
]


@dataclass
class Schedule:
    """Diffusion coefficients indexed so that entry t is step t, with t = 1..T.

    abar[0] = 1, alphas[t] = abar[t] / abar[t - 1], and betas[t] = 1 - alphas[t].
    """

    abar: jnp.ndarray
    alphas: jnp.ndarray
    T: int


def cosine_schedule(T: int, s: float = 0.008, alpha_floor: float = 1e-3) -> Schedule:
    """The cosine noise schedule the paper uses, with T_diff steps.

    alpha_t is floored the way the improved-DDPM schedule floors it, since the
    raw cosine sends alpha_T to zero and the 1 / sqrt(alpha_t) in Eq. (26) then
    blows up. abar is rebuilt as the cumulative product of the floored alphas so
    the forward marginal and the reverse step agree on the same numbers.
    """
    t = jnp.arange(T + 1, dtype=jnp.float32) / T
    f = jnp.cos((t + s) / (1.0 + s) * jnp.pi / 2) ** 2
    raw = f / f[0]
    alphas = jnp.clip(raw[1:] / raw[:-1], alpha_floor, 0.9999)
    alphas = jnp.concatenate([jnp.ones((1,)), alphas])
    return Schedule(abar=jnp.cumprod(alphas), alphas=alphas, T=T)


def q_sample(key, x0: jnp.ndarray, t: jnp.ndarray, sched: Schedule):
    """One draw from q(X^(t) | X^(0)), returning the noised sample and the noise."""
    eps = jax.random.normal(key, x0.shape)
    a = sched.abar[t]
    while a.ndim < x0.ndim:
        a = a[..., None]
    return jnp.sqrt(a) * x0 + jnp.sqrt(1.0 - a) * eps, eps


def _fourier(v: jnp.ndarray, dim: int, scale: float = 1.0) -> jnp.ndarray:
    freqs = jnp.exp(jnp.linspace(0.0, 4.0, dim // 2))
    ang = v[..., None] * freqs[None, :] * scale
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def init_eps_model(key, H: int, d: int, width: int = 256, depth: int = 3, emb: int = 32):
    """Parameters for the noise-prediction network eps_theta(X, t, y).

    ``null`` is the dedicated null-conditioning embedding that stands in for the
    dropped condition, which is what makes classifier-free guidance possible.
    """
    k_trunk, k_time, k_cond, k_null = jax.random.split(key, 4)
    return {
        "trunk": init_mlp(k_trunk, [H * d + 2 * emb] + [width] * depth + [H * d]),
        "time": init_mlp(k_time, [emb, emb, emb]),
        "cond": init_mlp(k_cond, [emb, emb, emb]),
        "null": jax.random.normal(k_null, (emb,)) * 0.1,
    }


def apply_eps_model(params, X: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray, drop: jnp.ndarray) -> jnp.ndarray:
    """eps_theta(X^(t), t, y), with ``drop`` selecting the null embedding per row."""
    emb = params["null"].shape[0]
    flat = X.reshape(X.shape[0], -1)
    t_emb = apply_mlp(params["time"], _fourier(t.astype(jnp.float32), emb, 0.02))
    y_emb = apply_mlp(params["cond"], _fourier(y.astype(jnp.float32), emb))
    y_emb = jnp.where(drop[:, None], params["null"][None, :], y_emb)
    out = apply_mlp(params["trunk"], jnp.concatenate([flat, t_emb, y_emb], axis=-1))
    return out.reshape(X.shape)


def guided_eps(params, X: jnp.ndarray, t: jnp.ndarray, y_tar: jnp.ndarray, s_cfg: float) -> jnp.ndarray:
    """Eq. (25), classifier-free guidance.

        eps_hat = eps(X, t, null) + s_cfg (eps(X, t, y_tar) - eps(X, t, null))

    At s_cfg = 1 this is the conditional prediction on its own, which the tests
    check exactly.
    """
    n = X.shape[0]
    uncond = apply_eps_model(params, X, t, y_tar, jnp.ones((n,), dtype=bool))
    cond = apply_eps_model(params, X, t, y_tar, jnp.zeros((n,), dtype=bool))
    return uncond + s_cfg * (cond - uncond)


def reverse_step(key, X: jnp.ndarray, t: int, eps_hat: jnp.ndarray, sched: Schedule) -> jnp.ndarray:
    """Eq. (26), one reverse step.

        X^(t-1) = (1 / sqrt(alpha_t)) (X^(t) - (1 - alpha_t) / sqrt(1 - abar_t) eps_hat)
                  + sigma_t z,  sigma_t^2 = (1 - abar_{t-1}) / (1 - abar_t) (1 - alpha_t)

    The noise term is dropped at the final reverse step, t = 1.
    """
    alpha = sched.alphas[t]
    abar_t = sched.abar[t]
    abar_prev = sched.abar[t - 1]
    mean = (X - (1.0 - alpha) / jnp.sqrt(1.0 - abar_t) * eps_hat) / jnp.sqrt(alpha)
    sigma = jnp.sqrt((1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - alpha))
    z = jnp.where(t > 1, 1.0, 0.0) * jax.random.normal(key, X.shape)
    return mean + sigma * z


def clamp_prediction(X: jnp.ndarray, t: int, eps_hat: jnp.ndarray, sched: Schedule, bound: float = 1.0) -> jnp.ndarray:
    """Re-express eps_hat so the x0 it implies stays inside [-bound, bound].

    Not in the paper. It is needed because y_tar = lambda_tar max y sits above
    every score the model was trained on, and guidance at that target pushes the
    iterate off the data range, after which every sample saturates at the same
    corner of the design box. Clamping the implied x0 is the usual fix and it
    leaves the form of Eq. (26) alone, since only eps_hat changes.
    """
    abar = sched.abar[t]
    x0 = (X - jnp.sqrt(1.0 - abar) * eps_hat) / jnp.sqrt(abar)
    x0 = jnp.clip(x0, -bound, bound)
    return (X - jnp.sqrt(abar) * x0) / jnp.sqrt(1.0 - abar)


@dataclass
class DiffusionModel:
    params: dict
    sched: Schedule
    horizon: int
    dim: int
    y_mean: float
    y_std: float

    def condition(self, y):
        """Scale a raw terminal score into the units the model was trained on."""
        return (jnp.asarray(y) - self.y_mean) / self.y_std


@partial(jax.jit, static_argnums=(4, 5, 6))
def _train(key, params, X0, y, steps, batch_size, T, p_drop, lr, ema_decay, sched_abar, sched_alphas):
    n = X0.shape[0]
    sched = Schedule(abar=sched_abar, alphas=sched_alphas, T=T)
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_fn(params, k):
        k_idx, k_t, k_eps, k_drop = jax.random.split(k, 4)
        idx = jax.random.randint(k_idx, (batch_size,), 0, n)
        t = jax.random.randint(k_t, (batch_size,), 1, T + 1)
        xt, eps = q_sample(k_eps, X0[idx], t, sched)
        drop = jax.random.uniform(k_drop, (batch_size,)) < p_drop
        pred = apply_eps_model(params, xt, t, y[idx], drop)
        return jnp.mean((pred - eps) ** 2)

    def step(carry, k):
        params, state, ema = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, k)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        ema = jax.tree.map(lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema, params)
        return (params, state, ema), loss

    (_, _, ema), losses = jax.lax.scan(step, (params, state, params), jax.random.split(key, steps))
    return ema, losses


def train_diffusion(
    key,
    designs: jnp.ndarray,
    y_end: jnp.ndarray,
    T: int = 100,
    steps: int = 3000,
    batch_size: int = 128,
    p_drop: float = 0.25,
    lr: float = 2e-4,
    width: int = 256,
    depth: int = 3,
    ema_decay: float = 0.995,
    y_scale: tuple[float, float] | None = None,
) -> DiffusionModel:
    """Fit eps_theta on D_traj with the terminal score as the condition.

    ``designs`` is (N, H, d) already scaled to roughly [-1, 1]; ``y_end`` is the
    terminal score y_H of each trajectory. The returned parameters are the EMA
    copy.

    ``y_scale`` is the (shift, scale) applied to the condition. Leaving it None
    standardizes against the retained endpoints, which is wrong when they are all
    near the top: y_tar = lambda_tar max y then lands tens of standard deviations
    out and the sampler collapses. The pipeline passes the offline score scale
    instead.
    """
    designs = jnp.asarray(designs, dtype=jnp.float32)
    y_end = jnp.asarray(y_end, dtype=jnp.float32)
    if y_scale is None:
        y_mean, y_std = float(y_end.mean()), float(y_end.std()) + 1e-6
    else:
        y_mean, y_std = float(y_scale[0]), float(y_scale[1])
    yn = (y_end - y_mean) / y_std
    k_init, k_train = jax.random.split(key)
    _, H, d = designs.shape
    params = init_eps_model(k_init, H, d, width=width, depth=depth)
    sched = cosine_schedule(T)
    ema, _ = _train(
        k_train, params, designs, yn, steps, batch_size, T, p_drop, lr, ema_decay, sched.abar, sched.alphas
    )
    return DiffusionModel(params=ema, sched=sched, horizon=H, dim=d, y_mean=y_mean, y_std=y_std)


def sample_trajectories(
    key,
    model: DiffusionModel,
    n: int,
    y_tar,
    context: jnp.ndarray | None = None,
    s_cfg: float = 1.5,
    clip_x0: bool = True,
) -> jnp.ndarray:
    """Lines 3 to 9 of Algorithm 6, the guided reverse process.

    ``context`` is (n, H_ctx, d). It is written into the first H_ctx positions
    before the loop, matching line 4, and again after every reverse step per
    Eq. (27), so the returned prefix equals the context exactly.
    """
    H, d = model.horizon, model.dim
    sched = model.sched
    y = jnp.broadcast_to(model.condition(y_tar), (n,))
    H_ctx = 0 if context is None else context.shape[1]

    k0, k_rest = jax.random.split(key)
    X = jax.random.normal(k0, (n, H, d))
    if H_ctx:
        X = X.at[:, :H_ctx].set(context)

    def body(carry, k):
        X, t = carry
        eps_hat = guided_eps(model.params, X, jnp.full((n,), t), y, s_cfg)
        if clip_x0:
            eps_hat = clamp_prediction(X, t, eps_hat, sched)
        X = reverse_step(k, X, t, eps_hat, sched)
        if H_ctx:
            X = X.at[:, :H_ctx].set(context)  # Eq. (27)
        return (X, t - 1), None

    (X, _), _ = jax.lax.scan(body, (X, sched.T), jax.random.split(k_rest, sched.T))
    return X
