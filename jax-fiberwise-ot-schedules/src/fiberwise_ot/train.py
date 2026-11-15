"""Tiny 1D MLP train loop and one-shot model-aware retrain."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from fiberwise_ot.allocate import (
    LAMBDA_DDPM,
    LAMBDA_FM,
    allocate_weights,
    phi_from_weights,
    apply_phi_inv,
)
from fiberwise_ot.paths import alphabar, cosine_phase_increment, cond_ot_arc_length
from fiberwise_ot.risk import ddpm_midpoint_risk, fm_midpoint_risk

HIDDEN = 32
WIDTHS = (2, HIDDEN, HIDDEN, 1)


def sample_mixture(key, n, means=(-2.0, 2.0), std=0.5):
    """Synthetic 1D two-component Gaussian mixture."""
    k1, k2 = jax.random.split(key)
    comp = jax.random.bernoulli(k1, 0.5, (n,))
    loc = jnp.where(comp, means[1], means[0])
    return loc + std * jax.random.normal(k2, (n,))


def mlp_init(key, widths=WIDTHS):
    params = []
    for din, dout in zip(widths[:-1], widths[1:]):
        key, k1, k2 = jax.random.split(key, 3)
        W = jax.random.normal(k1, (din, dout)) * jnp.sqrt(2.0 / din)
        b = jnp.zeros((dout,))
        params.append((W, b))
    return params


def mlp_apply(params, xt):
    h = xt
    for W, b in params[:-1]:
        h = jax.nn.silu(h @ W + b)
    W, b = params[-1]
    return jnp.squeeze(h @ W + b, axis=-1)


def _pair_xt(x, t):
    return jnp.stack([x, t], axis=-1)


def ddpm_loss(params, x0, eps, t, tau_edges=None, t_edges=None):
    if tau_edges is None:
        tau = t
    else:
        tau = apply_phi_inv(t, tau_edges, t_edges)
    ab = jnp.clip(alphabar(tau), 0.0, 1.0)
    alpha = jnp.sqrt(ab)
    sigma = jnp.sqrt(1.0 - ab)
    x = alpha * x0 + sigma * eps
    eps_hat = mlp_apply(params, _pair_xt(x, t))
    return jnp.mean(jnp.square(eps_hat - eps))


def fm_loss(params, x0, eps, t, tau_edges=None, t_edges=None):
    if tau_edges is None:
        tau = t
    else:
        tau = apply_phi_inv(t, tau_edges, t_edges)
    x = (1.0 - tau) * x0 + tau * eps
    target = eps - x0
    v_hat = mlp_apply(params, _pair_xt(x, t))
    return jnp.mean(jnp.square(v_hat - target))


def _sgd_step(params, grads, lr):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def train_loop(
    key,
    family="ddpm",
    steps=40,
    batch=64,
    lr=3e-3,
    tau_edges=None,
    t_edges=None,
    params=None,
):
    """Train a tiny 1D MLP. ``family`` is ``ddpm`` or ``fm``."""
    key, k_init = jax.random.split(key)
    if params is None:
        params = mlp_init(k_init)
    loss_fn = ddpm_loss if family == "ddpm" else fm_loss

    @jax.jit
    def step(params, key):
        k1, k2, k3 = jax.random.split(key, 3)
        x0 = sample_mixture(k1, batch)
        eps = jax.random.normal(k2, (batch,))
        t = jax.random.uniform(k3, (batch,))
        loss, grads = jax.value_and_grad(loss_fn)(
            params, x0, eps, t, tau_edges, t_edges
        )
        params = _sgd_step(params, grads, lr)
        return params, loss

    last = jnp.array(0.0)
    for i in range(steps):
        key, k = jax.random.split(key)
        params, last = step(params, k)
    return params, last


def _predict_from_params(params):
    def predict(x, t):
        return mlp_apply(params, _pair_xt(x, t))

    return predict


def estimate_interval_risk(params, key, family, n_intervals=16, n_samples=256):
    edges = jnp.linspace(0.0, 1.0, n_intervals + 1)
    k1, k2 = jax.random.split(key)
    x0 = sample_mixture(k1, n_samples)
    eps = jax.random.normal(k2, (n_samples,))
    predict = _predict_from_params(params)
    if family == "ddpm":
        r = ddpm_midpoint_risk(predict, x0, eps, edges)
        L = cosine_phase_increment(edges)
        lam = LAMBDA_DDPM
    else:
        r = fm_midpoint_risk(predict, x0, eps, edges)
        L = cond_ot_arc_length(edges)
        lam = LAMBDA_FM
    return L, r, lam, edges


def one_shot_retrain(
    key,
    family="ddpm",
    baseline_steps=40,
    retrain_steps=40,
    batch=64,
    n_intervals=16,
    n_risk_samples=256,
    lr=3e-3,
):
    """Train baseline, allocate from midpoint risk, retrain from scratch with Φ fixed."""
    key, k_base, k_risk, k_re = jax.random.split(key, 4)
    base_params, base_loss = train_loop(
        k_base, family=family, steps=baseline_steps, batch=batch, lr=lr
    )
    L, r, lam, _edges = estimate_interval_risk(
        base_params, k_risk, family, n_intervals=n_intervals, n_samples=n_risk_samples
    )
    w = allocate_weights(L, r, lam)
    tau_edges, t_edges = phi_from_weights(w)
    re_params, re_loss = train_loop(
        k_re,
        family=family,
        steps=retrain_steps,
        batch=batch,
        lr=lr,
        tau_edges=tau_edges,
        t_edges=t_edges,
    )
    return {
        "baseline_params": base_params,
        "retrain_params": re_params,
        "baseline_loss": base_loss,
        "retrain_loss": re_loss,
        "L": L,
        "r": r,
        "w": w,
        "lam": lam,
        "tau_edges": tau_edges,
        "t_edges": t_edges,
    }
