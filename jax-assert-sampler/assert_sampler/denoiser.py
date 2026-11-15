"""A small MLP epsilon-predictor for the 2D toy problem.

The network exists so the CIM noise model of Section 3.1 has real stored weights
to perturb. It is trained by regressing onto the analytic eps* of the Gaussian
mixture rather than onto sampled noise: eps* is the minimizer of the usual
denoising objective anyway, and the low-variance target gets the network close
enough to the exact score that whatever the samplers do differently under
hardware noise is not just leftover training error.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import optax

from .diffusion import (GaussianMixture, Schedule, analytic_eps, default_mixture,
                        linear_schedule)

TIME_FEATURES = 8


def init_params(key, in_dim: int = 2, width: int = 96, depth: int = 3):
    sizes = [in_dim + 2 * TIME_FEATURES] + [width] * (depth - 1) + [in_dim]
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        scale = jnp.sqrt(2.0 / n_in)
        params.append({"w": scale * jax.random.normal(k, (n_in, n_out)),
                       "b": jnp.zeros((n_out,))})
    return params


def time_features(t_norm: jnp.ndarray) -> jnp.ndarray:
    freqs = 2.0 ** jnp.arange(TIME_FEATURES) * jnp.pi
    ang = t_norm[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def apply(params, x: jnp.ndarray, t_norm: jnp.ndarray, *, key=None,
          sigma_y: float = 0.0) -> jnp.ndarray:
    """Forward pass. Every matmul is one CIM MVM, Equation 1.

    ``params`` is already the perturbed mapping W + eps_W when hardware noise is
    active. ``sigma_y`` is the output noise, redrawn per MVM, so it needs a key.
    """
    t = jnp.broadcast_to(t_norm, x.shape[:-1])
    h = jnp.concatenate([x, time_features(t)], axis=-1)
    keys = (jax.random.split(key, len(params)) if key is not None
            else [None] * len(params))
    for i, (layer, k) in enumerate(zip(params, keys)):
        h = h @ layer["w"]
        if sigma_y > 0.0 and k is not None:
            h = h + sigma_y * jax.random.normal(k, h.shape)
        h = h + layer["b"]
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h


def _loss(params, mix, sched, x, t_idx):
    ab = sched.alphas_cumprod[t_idx][:, None]
    target = jax.vmap(analytic_eps, in_axes=(None, 0, 0))(
        mix, x[:, None, :], ab[:, 0])[:, 0, :]
    pred = apply(params, x, t_idx / sched.num_train_steps)
    return jnp.mean((pred - target) ** 2)


def train(key, mix: GaussianMixture, sched: Schedule, *, steps: int = 4000,
          batch: int = 1024, width: int = 96, depth: int = 3, lr: float = 3e-3):
    """Fit the epsilon-predictor over the whole (x_t, t) support the sampler visits."""
    k_init, key = jax.random.split(key)
    params = init_params(k_init, in_dim=mix.means.shape[1], width=width, depth=depth)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        k_x, k_t, k_e = jax.random.split(key, 3)
        x0 = mix.sample(k_x, batch)
        t_idx = jax.random.randint(k_t, (batch,), 0, sched.num_train_steps)
        ab = sched.alphas_cumprod[t_idx][:, None]
        eps = jax.random.normal(k_e, x0.shape)
        xt = jnp.sqrt(ab) * x0 + jnp.sqrt(1.0 - ab) * eps
        loss, grads = jax.value_and_grad(_loss)(params, mix, sched, xt, t_idx)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, loss

    loss = jnp.array(0.0)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        params, state, loss = step(params, state, sub)
    return params, float(loss)


@functools.lru_cache(maxsize=4)
def pretrained(seed: int = 0, steps: int = 4000):
    """Train once per process and reuse. Deterministic in ``seed``."""
    mix = default_mixture()
    sched = linear_schedule()
    params, loss = train(jax.random.key(seed), mix, sched, steps=steps)
    return params, mix, sched, loss
