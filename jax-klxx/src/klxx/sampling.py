"""Quench-and-temper (Alg. 2), biased SMC (Alg. 3), MALA, and the xi mixture."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def quench(y, U, n_steps=30, lr=0.02):
    """Gradient descent on U. y has shape (n, d). U is a scalar map (d,) -> ()."""

    def body(_i, y):
        g = jax.vmap(jax.grad(U))(y)
        return y - lr * g

    return jax.lax.fori_loop(0, n_steps, body, y)


def mala_step(key, y, U, step_size):
    """One vectorized MALA step targeting pi ∝ exp(-U). y: (n, d)."""
    g = jax.vmap(jax.grad(U))(y)
    key, k_noise, k_u = jax.random.split(key, 3)
    noise = jax.random.normal(k_noise, y.shape)
    y_prop = y - step_size * g + jnp.sqrt(2.0 * step_size) * noise
    g_prop = jax.vmap(jax.grad(U))(y_prop)

    def log_q(yp, yc, gc):
        return -jnp.sum((yp - yc + step_size * gc) ** 2, axis=-1) / (4.0 * step_size)

    u_y = jax.vmap(U)(y)
    u_p = jax.vmap(U)(y_prop)
    log_alpha = -u_p + u_y + log_q(y, y_prop, g_prop) - log_q(y_prop, y, g)
    accept = jnp.log(jax.random.uniform(k_u, (y.shape[0],))) < log_alpha
    y_new = jnp.where(accept[:, None], y_prop, y)
    return y_new, key


def mala(y, U, key, n_steps=10, step_size=0.02):
    """Short MALA chain on U."""

    def body(_i, carry):
        y, key = carry
        return mala_step(key, y, U, step_size)

    y, key = jax.lax.fori_loop(0, n_steps, body, (y, key))
    return y


def qt(
    y,
    U,
    key,
    melt=2.0,
    quench_steps=30,
    quench_lr=0.02,
    mala_steps=10,
    mala_step_size=0.02,
):
    """Melt, quench, temper (Alg. 2). Quench is GD rather than L-BFGS."""
    key, k_melt, k_mala = jax.random.split(key, 3)
    y = y + melt * jax.random.normal(k_melt, y.shape)
    y = quench(y, U, n_steps=quench_steps, lr=quench_lr)
    y = mala(y, U, k_mala, n_steps=mala_steps, step_size=mala_step_size)
    return y


def mix_xi(qt_samples, nu_samples, key, alpha=0.5, beta=0.5):
    """Mixture xi = alpha * pi-hat + beta * nu. alpha + beta should be 1.

    Each row is taken from qt_samples with probability alpha, else from nu_samples.
    alpha=1, beta=0 returns the QT batch. alpha=0, beta=1 returns the pushforward batch.
    """
    del beta
    n = qt_samples.shape[0]
    pick_qt = jax.random.uniform(key, (n,)) < alpha
    return jnp.where(pick_qt[:, None], qt_samples, nu_samples)


def smc(y, z_fn, U, key, M=4, mala_steps=5, mala_step_size=0.02):
    """Biased target surrogate (Alg. 3).

    For m = 1..M: weight by exp(z/M), resample, short MALA on pi (i.e. on U).
    MALA targets pi rather than the bridge mu_m, matching the paper.
    """

    def one_level(_m, carry):
        y, key = carry
        z = z_fn(y)
        logw = z / M
        key, k_rs, k_mala = jax.random.split(key, 3)
        idx = jax.random.choice(k_rs, y.shape[0], shape=(y.shape[0],), p=jax.nn.softmax(logw))
        y = y[idx]
        y = mala(y, U, k_mala, n_steps=mala_steps, step_size=mala_step_size)
        return y, key

    y, key = jax.lax.fori_loop(0, M, one_level, (y, key))
    return y


def sample_two_mode(key, n, sigma=0.3):
    """iid samples from the equal-weight two-mode GMM at (+/-2, 0)."""
    k1, k2 = jax.random.split(key)
    right = jax.random.bernoulli(k1, 0.5, (n,))
    means = jnp.where(right[:, None], jnp.array([2.0, 0.0]), jnp.array([-2.0, 0.0]))
    return means + sigma * jax.random.normal(k2, (n, 2))
