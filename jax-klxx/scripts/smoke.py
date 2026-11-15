"""Train a small 2D coupling flow with KLXX vs forward KL on the two-mode GMM."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from klxx.flow import forward, init_coupling, inverse, source_potential, two_mode_potential
from klxx.loss import klxx_loss, z_from_flow
from klxx.sampling import mix_xi, qt, sample_two_mode


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(grads, params, state, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state["v"], grads)
    mhat = jax.tree.map(lambda m: m / (1.0 - b1 ** t), m)
    vhat = jax.tree.map(lambda v: v / (1.0 - b2 ** t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


def mode_coverage(params, key, n=256, radius=1.0):
    x = jax.random.normal(key, (n, 2))
    y, _ = inverse(params, x)
    modes = jnp.array([[-2.0, 0.0], [2.0, 0.0]])
    hits = 0
    for m in modes:
        if bool(jnp.any(jnp.linalg.norm(y - m, axis=-1) < radius)):
            hits += 1
    return hits


def train(lam, theta, key, n_layers=6, hidden=32, batch=64, steps=40, lr=0.02):
    key, k_init, k_qt = jax.random.split(key, 3)
    params = init_coupling(k_init, dim=2, n_layers=n_layers, hidden=hidden)
    x_seed = jax.random.normal(k_qt, (batch, 2))
    y_qt = qt(x_seed, two_mode_potential, k_qt, melt=2.5, quench_steps=20, mala_steps=5)
    y_qt = jax.lax.stop_gradient(y_qt)

    def loss_fn(p, y_pi, y_xi):
        z_pi = z_from_flow(p, y_pi, two_mode_potential, source_potential, forward)
        z_xi = z_from_flow(p, y_xi, two_mode_potential, source_potential, forward)
        return klxx_loss(z_pi, z_xi, lam=lam, theta=theta)

    @jax.jit
    def step(params, state, y_pi, y_xi):
        loss, grads = jax.value_and_grad(loss_fn)(params, y_pi, y_xi)
        params, state = adam_update(grads, params, state, lr=lr)
        return params, state, loss

    key, k_pi, k_nu, k_mix = jax.random.split(key, 4)
    y_pi = sample_two_mode(k_pi, batch)
    x0 = jax.random.normal(k_nu, (batch, 2))
    y_nu, _ = inverse(params, x0)
    y_xi = mix_xi(y_qt, jax.lax.stop_gradient(y_nu), k_mix, alpha=0.5, beta=0.5)
    loss_before = float(loss_fn(params, y_pi, y_xi))

    state = adam_init(params)
    loss = jnp.array(loss_before)
    for _ in range(steps):
        key, k_pi, k_nu, k_mix = jax.random.split(key, 4)
        y_pi = sample_two_mode(k_pi, batch)
        x0 = jax.random.normal(k_nu, (batch, 2))
        y_nu, _ = inverse(params, x0)
        y_xi = mix_xi(y_qt, jax.lax.stop_gradient(y_nu), k_mix, alpha=0.5, beta=0.5)
        params, state, loss = step(params, state, y_pi, y_xi)

    loss_after = float(loss)
    key, k_cov = jax.random.split(key)
    cov = mode_coverage(params, k_cov)
    return loss_before, loss_after, cov


def main():
    key = jax.random.PRNGKey(0)
    key, k_klxx, k_fkl = jax.random.split(key, 3)
    loss_before, loss_after, cov = train(1.0, 1.0, k_klxx)
    fkl_before, fkl_after, fkl_cov = train(0.0, 0.0, k_fkl)
    print(f"loss_before={loss_before:.6f}")
    print(f"loss_after={loss_after:.6f}")
    print(f"mode_coverage={cov}")
    print(f"fkl_loss_before={fkl_before:.6f}")
    print(f"fkl_loss_after={fkl_after:.6f}")
    print(f"fkl_mode_coverage={fkl_cov}")
    finite = jnp.isfinite(jnp.array(loss_before)) and jnp.isfinite(jnp.array(loss_after))
    dropped = loss_after < loss_before
    if not (bool(finite) and dropped):
        print("smoke failed: KLXX loss not finite or did not drop", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
