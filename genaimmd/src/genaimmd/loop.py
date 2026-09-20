"""GenAIMMD cycle (Fig. 1 / Sec. II E) and a tiny Adam trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from genaimmd.bg import bg_loss, integer_bias_centers, target_energy
from genaimmd.committor import aimmd_loss, init_committor, pB, q
from genaimmd.dynamics import integrate, two_way_shoot
from genaimmd.flow import forward, init_realnvp
from genaimmd.potential import KBIAS, MIN_A, MIN_B


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(
    params,
    grads,
    state,
    lr: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(
        lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1.0 - b2) * jnp.square(g), state["v"], grads
    )
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": t}


def init_samples(key, n_a: int = 100, n_b: int = 100, n_path: int = 100, noise: float = 0.05):
    k1, k2, k3 = jax.random.split(key, 3)
    xa = MIN_A + noise * jax.random.normal(k1, (n_a, 2))
    xb = MIN_B + noise * jax.random.normal(k2, (n_b, 2))
    t = jnp.linspace(0.0, 1.0, n_path)[:, None]
    xp = (1.0 - t) * MIN_A + t * MIN_B + noise * jax.random.normal(k3, (n_path, 2))
    return jnp.concatenate([xa, xb, xp], axis=0)


def init_state(key, n_a: int = 100, n_b: int = 100, n_path: int = 100):
    k_s, k_c, k_f = jax.random.split(key, 3)
    samples = init_samples(k_s, n_a=n_a, n_b=n_b, n_path=n_path)
    return {
        "samples": samples,
        "n_initial": int(samples.shape[0]),
        "committor_params": init_committor(k_c),
        "flow_params": init_realnvp(k_f),
        "bias_centers": jnp.full((samples.shape[0],), 0.5),
        "committor_loss": jnp.array(0.0),
        "bg_loss": jnp.array(0.0),
    }


def _shoot_batch(samples, key, max_steps: int):
    keys = jax.random.split(key, samples.shape[0])

    def one(x, k):
        out = two_way_shoot(x, k, max_steps=max_steps)
        return out["labels"], out["hab"]

    labels, hab = jax.vmap(one)(samples, keys)
    return labels, hab


def train_committor(params, x, s, key, n_steps: int = 200, batch_size: int = 100, lr: float = 1e-4):
    n = x.shape[0]
    batch_size = min(batch_size, n)
    opt = adam_init(params)
    loss = jnp.array(0.0)

    def step(i, carry):
        params, opt, key, loss = carry
        key, kb = jax.random.split(key)
        idx = jax.random.choice(kb, n, (batch_size,), replace=False)
        xb = x[idx]
        sb = s[idx]

        def loss_fn(p):
            return aimmd_loss(p, xb, sb) / batch_size

        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt = adam_update(params, grads, opt, lr=lr)
        return params, opt, key, loss

    params, opt, key, loss = jax.lax.fori_loop(0, n_steps, step, (params, opt, key, loss))
    return params, loss


def train_bg(
    flow_params,
    x,
    c,
    committor_params,
    key,
    n_steps: int = 200,
    batch_size: int = 100,
    lr: float = 1e-4,
):
    n = x.shape[0]
    batch_size = min(batch_size, n)
    opt = adam_init(flow_params)
    loss = jnp.array(0.0)

    def step(i, carry):
        flow_params, opt, key, loss = carry
        key, k1, k2, k3 = jax.random.split(key, 4)
        idx = jax.random.choice(k1, n, (batch_size,), replace=False)
        xb = x[idx]
        cb = c[idx]
        z = jax.random.normal(k2, (batch_size, 2))
        c_z = c[jax.random.choice(k3, n, (batch_size,), replace=True)]

        def loss_fn(p):
            return bg_loss(p, z, xb, c_z, cb, committor_params)

        loss, grads = jax.value_and_grad(loss_fn)(flow_params)
        flow_params, opt = adam_update(flow_params, grads, opt, lr=lr)
        return flow_params, opt, key, loss

    flow_params, opt, key, loss = jax.lax.fori_loop(
        0, n_steps, step, (flow_params, opt, key, loss)
    )
    return flow_params, loss


def relax_biased(samples, bias_centers, committor_params, key, n_steps: int, kbias: float = KBIAS):
    keys = jax.random.split(key, samples.shape[0])

    def one(x, c, k):
        def force_fn(xx):
            return -jax.grad(lambda y: target_energy(y, c, committor_params, kbias))(xx)

        x_end, _ = integrate(x, k, n_steps, force_fn=force_fn)
        return x_end

    return jax.vmap(one)(samples, bias_centers, keys)


def generate_samples(flow_params, bias_centers, key, n: int):
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (n, 2))
    idx = jax.random.choice(k2, bias_centers.shape[0], (n,), replace=True)
    c = bias_centers[idx]
    x, _ = forward(flow_params, z, c)
    return x, c


def fifo_push(old, new, max_size: int):
    cat = jnp.concatenate([old, new], axis=0)
    return cat[-max_size:]


def run_cycle(
    state,
    key,
    n_committor_steps: int = 200,
    n_bg_steps: int = 200,
    n_relax: int = 500,
    max_shoot_steps: int = 500,
    batch_size: int = 100,
    lr: float = 1e-4,
    kbias: float = KBIAS,
):
    """One GenAIMMD iteration: shoot, train q, relabel, relax, train BG, generate."""
    k_shoot, k_q, k_rel, k_bg, k_gen = jax.random.split(key, 5)
    samples = state["samples"]
    labels, _hab = _shoot_batch(samples, k_shoot, max_shoot_steps)
    q_params, q_loss = train_committor(
        state["committor_params"],
        samples,
        labels,
        k_q,
        n_steps=n_committor_steps,
        batch_size=batch_size,
        lr=lr,
    )
    centers = integer_bias_centers(q(q_params, samples))
    samples = relax_biased(samples, centers, q_params, k_rel, n_relax, kbias=kbias)
    flow_params, bg_l = train_bg(
        state["flow_params"],
        samples,
        centers,
        q_params,
        k_bg,
        n_steps=n_bg_steps,
        batch_size=batch_size,
        lr=lr,
    )
    n_init = state["n_initial"]
    fifo_size = 2 * n_init
    n_new = n_init
    x_new, c_new = generate_samples(flow_params, centers, k_gen, n_new)
    samples = fifo_push(samples, x_new, fifo_size)
    centers = fifo_push(centers, c_new, fifo_size)
    return {
        "samples": samples,
        "n_initial": n_init,
        "committor_params": q_params,
        "flow_params": flow_params,
        "bias_centers": centers,
        "committor_loss": q_loss,
        "bg_loss": bg_l,
        "pB": pB(q_params, samples),
    }


def run_loop(
    key,
    n_cycles: int = 2,
    n_a: int = 100,
    n_b: int = 100,
    n_path: int = 100,
    **cycle_kwargs,
):
    key, k0 = jax.random.split(key)
    state = init_state(k0, n_a=n_a, n_b=n_b, n_path=n_path)
    for _ in range(n_cycles):
        key, k = jax.random.split(key)
        state = run_cycle(state, k, **cycle_kwargs)
    return state
