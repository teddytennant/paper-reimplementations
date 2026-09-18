"""Tiny Soft Actor-Critic in JAX (actor, twin critics, entropy temperature, replay).

A compact reimplementation of Haarnoja et al. (2018) sufficient for the
toy environment. Not the Stable-Baselines3 / MuJoCo setup used in the paper.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def _init_mlp(key, sizes, last_scale=0.01):
    params = []
    n_layers = len(sizes) - 1
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        std = last_scale if i == n_layers - 1 else (1.0 / np.sqrt(din))
        w = jax.random.normal(k, (din, dout), dtype=jnp.float32) * std
        b = jnp.zeros((dout,), dtype=jnp.float32)
        params.append({"w": w, "b": b})
    return params


def _mlp(params, x):
    for i, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def _actor_forward(params, obs):
    out = _mlp(params, obs)
    mean, log_std = jnp.split(out, 2, axis=-1)
    log_std = jnp.clip(log_std, -20.0, 2.0)
    return mean, log_std


def _sample_action(params, obs, key):
    mean, log_std = _actor_forward(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    raw = mean + std * noise
    action = jnp.tanh(raw)
    log_prob = _tanh_gaussian_log_prob(raw, action, mean, log_std)
    return action, log_prob


def _tanh_gaussian_log_prob(raw, action, mean, log_std):
    var = jnp.exp(2.0 * log_std)
    log_gauss = -0.5 * (
        ((raw - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    log_gauss = jnp.sum(log_gauss, axis=-1)
    log_det = jnp.sum(jnp.log(1.0 - action**2 + 1e-6), axis=-1)
    return log_gauss - log_det


def _q_forward(params, obs, action):
    x = jnp.concatenate([obs, action], axis=-1)
    return jnp.squeeze(_mlp(params, x), axis=-1)


def _adam_init(params):
    return {
        "m": jax.tree.map(jnp.zeros_like, params),
        "v": jax.tree.map(jnp.zeros_like, params),
        "t": np.array(0, dtype=np.int32),
    }


def _adam_step(params, grads, state, lr):
    t = int(state["t"]) + 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state["v"], grads)
    mhat = jax.tree.map(lambda m: m / (1.0 - b1**t), m)
    vhat = jax.tree.map(lambda v: v / (1.0 - b2**t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": np.array(t, dtype=np.int32)}


def _soft_update(target, source, tau):
    return jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = int(capacity)
        self.idx = 0
        self.size = 0
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.act[i] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rew[i] = float(reward)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "action": self.act[idx],
            "reward": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }


class SAC:
    """Actor, twin critics with Polyak targets, learnable temperature, replay."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        key=None,
        hidden=32,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        replay_capacity=10_000,
        batch_size=64,
        target_entropy=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else -float(self.action_dim)
        )
        self.replay = ReplayBuffer(replay_capacity, self.obs_dim, self.action_dim)
        self._np_rng = np.random.default_rng(0)

        key, k_actor, k_q1, k_q2 = jax.random.split(key, 4)
        self.key = key
        sizes_actor = [self.obs_dim, self.hidden, self.hidden, 2 * self.action_dim]
        sizes_q = [self.obs_dim + self.action_dim, self.hidden, self.hidden, 1]
        self.actor = _init_mlp(k_actor, sizes_actor)
        self.q1 = _init_mlp(k_q1, sizes_q)
        self.q2 = _init_mlp(k_q2, sizes_q)
        self.q1_target = jax.tree.map(lambda x: x.copy(), self.q1)
        self.q2_target = jax.tree.map(lambda x: x.copy(), self.q2)
        self.log_alpha = jnp.array(0.0, dtype=jnp.float32)

        self.opt_actor = _adam_init(self.actor)
        self.opt_q1 = _adam_init(self.q1)
        self.opt_q2 = _adam_init(self.q2)
        self.opt_alpha = _adam_init(self.log_alpha)

    def _split(self):
        self.key, sub = jax.random.split(self.key)
        return sub

    def act(self, obs, deterministic=True):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[None, :]
        mean, log_std = _actor_forward(self.actor, obs)
        if deterministic:
            action = jnp.tanh(mean)
        else:
            action, _ = _sample_action(self.actor, obs, self._split())
        action = np.asarray(action)
        if squeeze:
            action = action[0]
        return action

    def store(self, obs, action, reward, next_obs, done):
        self.replay.add(obs, action, reward, next_obs, done)

    def update(self, batch=None):
        """One SAC step. Returns True if an update ran."""
        if batch is None:
            if self.replay.size < self.batch_size:
                return False
            batch = self.replay.sample(self.batch_size, self._np_rng)
        obs = jnp.asarray(batch["obs"], dtype=jnp.float32)
        action = jnp.asarray(batch["action"], dtype=jnp.float32)
        reward = jnp.asarray(batch["reward"], dtype=jnp.float32)
        next_obs = jnp.asarray(batch["next_obs"], dtype=jnp.float32)
        done = jnp.asarray(batch["done"], dtype=jnp.float32)

        key = self._split()
        next_action, next_logp = _sample_action(self.actor, next_obs, key)
        q1_t = _q_forward(self.q1_target, next_obs, next_action)
        q2_t = _q_forward(self.q2_target, next_obs, next_action)
        q_t = jnp.minimum(q1_t, q2_t)
        alpha = jnp.exp(self.log_alpha)
        target = reward + self.gamma * (1.0 - done) * (q_t - alpha * next_logp)
        target = jax.lax.stop_gradient(target)

        def q1_loss(p):
            q = _q_forward(p, obs, action)
            return jnp.mean((q - target) ** 2)

        def q2_loss(p):
            q = _q_forward(p, obs, action)
            return jnp.mean((q - target) ** 2)

        g_q1 = jax.grad(q1_loss)(self.q1)
        g_q2 = jax.grad(q2_loss)(self.q2)
        self.q1, self.opt_q1 = _adam_step(self.q1, g_q1, self.opt_q1, self.lr)
        self.q2, self.opt_q2 = _adam_step(self.q2, g_q2, self.opt_q2, self.lr)

        key = self._split()

        def actor_loss(p):
            a, logp = _sample_action(p, obs, key)
            q = jnp.minimum(_q_forward(self.q1, obs, a), _q_forward(self.q2, obs, a))
            return jnp.mean(alpha * logp - q)

        g_actor = jax.grad(actor_loss)(self.actor)
        self.actor, self.opt_actor = _adam_step(self.actor, g_actor, self.opt_actor, self.lr)

        key = self._split()
        _, logp_pi = _sample_action(self.actor, obs, key)
        logp_pi = jax.lax.stop_gradient(logp_pi)

        def alpha_loss(log_alpha):
            return -jnp.mean(log_alpha * (logp_pi + self.target_entropy))

        g_alpha = jax.grad(alpha_loss)(self.log_alpha)
        self.log_alpha, self.opt_alpha = _adam_step(
            self.log_alpha, g_alpha, self.opt_alpha, self.lr
        )

        self.q1_target = _soft_update(self.q1_target, self.q1, self.tau)
        self.q2_target = _soft_update(self.q2_target, self.q2, self.tau)
        return True
