"""Affine transport, importance weight, tail penalty, and training."""

from __future__ import annotations

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()

import jax
import jax.numpy as jnp
import optax


def init_affine(d):
    """L starts at I (raw diagonal 0, exp(0)=1) and b starts at 0."""
    n_tril = d * (d + 1) // 2
    return {
        "raw_tril": jnp.zeros((n_tril,)),
        "b": jnp.zeros((d,)),
    }


def lower_triangular(raw_tril, d):
    """Lower triangular L with positive diagonal exp(raw_ii)."""
    raw_tril = jnp.asarray(raw_tril)
    L = jnp.zeros((d, d), dtype=raw_tril.dtype).at[jnp.tril_indices(d)].set(raw_tril)
    L = L.at[jnp.diag_indices(d)].set(jnp.exp(jnp.diag(L)))
    return L


def affine_log_det(L):
    """log|det| = sum log diag."""
    return jnp.sum(jnp.log(jnp.diag(L)))


def affine_apply(z, params):
    """T(z) = L z + b."""
    z = jnp.asarray(z)
    d = z.shape[-1]
    L = lower_triangular(params["raw_tril"], d)
    x = z @ L.T + params["b"]
    return x, affine_log_det(L), L


def log_w(z, log_pi_tz, log_abs_det):
    """log w(z) = log pi(T(z)) + log|det grad T(z)| + 0.5*||z||**2.

    The additive constant in log rho for rho = N(0, I) is omitted.
    """
    z = jnp.asarray(z)
    return log_pi_tz + log_abs_det + 0.5 * jnp.sum(jnp.square(z), axis=-1)


def log_abs_det_jacobian(transform, z):
    z = jnp.asarray(z)

    def one(z1):
        return jnp.linalg.slogdet(jax.jacobian(transform)(z1))[1]

    if z.ndim == 1:
        return one(z)
    return jax.vmap(one)(z)


def log_w_transform(z, transform, log_pi):
    """log w for a general map, with log|det grad T| from the Jacobian."""
    z = jnp.asarray(z)
    if z.ndim == 1:
        x = transform(z)
        log_pi_x = log_pi(x)
    else:
        x = jax.vmap(transform)(z)
        log_pi_x = jax.vmap(log_pi)(x)
    return log_w(z, log_pi_x, log_abs_det_jacobian(transform, z))


def log_w_affine(z, params, log_pi):
    z = jnp.asarray(z)
    x, log_det, _ = affine_apply(z, params)
    if x.ndim == 1:
        log_pi_x = log_pi(x)
    else:
        log_pi_x = jax.vmap(log_pi)(x)
    return log_w(z, log_pi_x, log_det)


def pushforward_gaussian_log_prob(L, b):
    """Normalized log density of X = L Z + b for Z ~ N(0, I). L is triangular."""
    L = jnp.asarray(L)
    b = jnp.asarray(b)
    d = b.shape[0]
    log_det = affine_log_det(L)
    const = -0.5 * d * jnp.log(2.0 * jnp.pi) - log_det

    def log_pi(x):
        z = jnp.linalg.solve(L, x - b)
        return const - 0.5 * jnp.sum(jnp.square(z), axis=-1)

    return log_pi


def gaussian_log_prob(mu, cov):
    """Normalized log density of N(mu, cov)."""
    mu = jnp.asarray(mu)
    cov = jnp.asarray(cov)
    d = mu.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    _, logdet = jnp.linalg.slogdet(cov)
    const = -0.5 * d * jnp.log(2.0 * jnp.pi) - 0.5 * logdet

    def log_pi(x):
        diff = x - mu
        quad = jnp.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        return const - 0.5 * quad

    return log_pi


def sample_tail(key, d, n):
    """zeta = R*u, R uniform on [sqrt(d)+1, sqrt(d)+5], u on the sphere."""
    key_r, key_g = jax.random.split(key)
    lo = jnp.sqrt(d) + 1.0
    hi = jnp.sqrt(d) + 5.0
    radius = jax.random.uniform(key_r, (n,), minval=lo, maxval=hi)
    gaussian = jax.random.normal(key_g, (n, d))
    direction = gaussian / jnp.linalg.norm(gaussian, axis=-1, keepdims=True)
    return radius[:, None] * direction


def tail_penalty(log_w_zeta, mean_log_w, c):
    """E[max(0, log w(zeta) - mean_log_w - c)**2]. Training penalty, not a theorem."""
    excess = jnp.maximum(0.0, log_w_zeta - mean_log_w - c)
    return jnp.mean(jnp.square(excess))


def objective(params, z, zeta, log_pi, tail_weight, c):
    """-mean(log w) plus a tail penalty. mean_log_w is the rho-batch mean."""
    log_w_z = log_w_affine(z, params, log_pi)
    mean_log_w = jnp.mean(log_w_z)
    log_w_zeta = log_w_affine(zeta, params, log_pi)
    pen = tail_penalty(log_w_zeta, jax.lax.stop_gradient(mean_log_w), c)
    return -mean_log_w + tail_weight * pen


def train_affine(
    key,
    log_pi,
    d,
    n_steps,
    batch,
    n_tail,
    tail_weight,
    c,
    lr,
    params=None,
):
    """Minimize -mean(log w) plus a small tail penalty for an affine map."""
    if params is None:
        params = init_affine(d)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, z, zeta):
        return objective(params, z, zeta, log_pi, tail_weight, c)

    def step(carry, _):
        params, opt_state, key = carry
        key, key_z, key_t = jax.random.split(key, 3)
        z = jax.random.normal(key_z, (batch, d))
        zeta = sample_tail(key_t, d, n_tail)
        loss, grads = jax.value_and_grad(loss_fn)(params, z, zeta)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key), loss

    (params, _, _), losses = jax.lax.scan(
        step, (params, opt_state, key), xs=None, length=n_steps
    )
    return params, losses
