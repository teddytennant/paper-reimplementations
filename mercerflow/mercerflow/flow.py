"""Conditional flow matching in coefficient space, Section 2.3."""

import jax
import jax.numpy as jnp
import optax

from mercerflow.codecs import decode, encode
from mercerflow.kernels import sample_conditional_future
from mercerflow.model import apply_velocity_mlp


def interpolant(c0, c1, t):
    """Linear conditional path and target velocity, equation (9).

    ct = (1 - t) c0 + t c1, u = c1 - c0.
    c0 and c1 have shape (L,) or (batch, L). t is a scalar or (batch,).
    """
    c0 = jnp.asarray(c0)
    c1 = jnp.asarray(c1)
    t = jnp.asarray(t, dtype=c0.dtype)
    if c0.ndim == 1:
        ct = (1.0 - t) * c0 + t * c1
    else:
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (c0.shape[0],))
        ct = (1.0 - t)[:, None] * c0 + t[:, None] * c1
    u = c1 - c0
    return ct, u


def shared_context_velocity(Phi, y0, y_future):
    """u = Phi.T @ concat(0, y_future - y0).

    The observed past is shared by the source and target windows, so the
    ambient displacement is zero on the past block (Section 2.3).
    """
    delta_future = y_future - y0
    n_future = delta_future.shape[-1]
    n_past = Phi.shape[0] - n_future
    if n_past < 0:
        raise ValueError("future is longer than the basis")
    zeros = jnp.zeros(delta_future.shape[:-1] + (n_past,), dtype=delta_future.dtype)
    ambient = jnp.concatenate([zeros, delta_future], axis=-1)
    return encode(Phi, ambient)


def encode_endpoints(Phi, y_past, y0, y_future):
    """c0 = Phi.T @ concat(yp, y0), c1 = Phi.T @ concat(yp, yf)."""
    source = jnp.concatenate([y_past, y0], axis=-1)
    target = jnp.concatenate([y_past, y_future], axis=-1)
    return encode(Phi, source), encode(Phi, target)


def flow_matching_loss(params, c0, c1, t):
    """Conditional flow-matching loss, equation (10).

    Mean over the batch of ||v_theta(t, ct) - u||^2. A single window
    returns the squared Euclidean norm itself.
    """
    t = jnp.asarray(t, dtype=jnp.asarray(c0).dtype)
    ct, u = interpolant(c0, c1, t)
    if c0.ndim == 2 and t.ndim == 0:
        t = jnp.broadcast_to(t, (c0.shape[0],))
    v = apply_velocity_mlp(params, t, ct)
    residual = v - u
    if residual.ndim == 1:
        return jnp.sum(residual ** 2)
    return jnp.mean(jnp.sum(residual ** 2, axis=-1))


def train_step(params, opt_state, optimizer, c0, c1, t):
    """One optimizer step on a fixed coefficient batch.

    Returns updated params, updated optimizer state, and the loss before
    the step.
    """
    loss, grads = jax.value_and_grad(flow_matching_loss)(params, c0, c1, t)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def sample_cfm_batch(key, Phi, K, y_past, y_future):
    """Sample source noise, encode both endpoints, and draw flow time.

    y_past and y_future are batches of shape (B, Lp) and (B, Lf).
    t is drawn independently from Uniform[0, 1] for each row.
    """
    key_noise, key_time = jax.random.split(key)
    n_past = y_past.shape[-1]
    y0 = sample_conditional_future(key_noise, K, y_past, n_past)
    c0, c1 = encode_endpoints(Phi, y_past, y0, y_future)
    t = jax.random.uniform(key_time, (y_past.shape[0],), dtype=c0.dtype)
    return c0, c1, t


def training_step(params, opt_state, optimizer, Phi, K, y_past, y_future, key):
    """One conditional flow-matching update on synthetic or real windows.

    Samples y0 from equation (3), builds the equation (9) path in
    coefficient space, and takes one step on equation (10).
    """
    y_past = jnp.asarray(y_past)
    y_future = jnp.asarray(y_future)
    if y_past.ndim == 1:
        y_past = y_past[None, :]
        y_future = y_future[None, :]
    c0, c1, t = sample_cfm_batch(key, Phi, K, y_past, y_future)
    return train_step(params, opt_state, optimizer, c0, c1, t)


def euler_integrate(params, c0, n_steps=20):
    """Fixed-step forward Euler from t = 0 to t = 1.

    c_{k+1} = c_k + dt * v_theta(t_k, c_k), t_k = k / n_steps,
    dt = 1 / n_steps, k = 0, ..., n_steps - 1. Default is 20 steps.
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    dt = 1.0 / n_steps
    c = c0
    for step in range(n_steps):
        t = step * dt
        c = c + dt * apply_velocity_mlp(params, t, c)
    return c


def euler_forecast(params, Phi, y_past, K, key, n_steps=20):
    """Sample a conditional prior, integrate, and return the future slice.

    y_past has shape (Lp,) or (batch, Lp). A single series returns shape
    (Lf,). The default integrator uses 20 Euler steps.
    """
    y_past = jnp.asarray(y_past)
    n_past = int(y_past.shape[-1])
    length = int(Phi.shape[0])
    if K.shape != (length, length):
        raise ValueError("K must be (L, L) with L = Phi.shape[0]")
    y0 = sample_conditional_future(key, K, y_past, n_past)
    source = jnp.concatenate([y_past, y0], axis=-1)
    c0 = encode(Phi, source)
    c_hat = euler_integrate(params, c0, n_steps=n_steps)
    y_hat = decode(Phi, c_hat)
    return y_hat[..., n_past:]
