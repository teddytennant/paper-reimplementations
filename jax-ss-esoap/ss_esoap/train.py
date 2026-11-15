"""A 1D Poisson PINN, small enough to run on a CPU in a few seconds.

    -u''(x) = f(x)  on (0, 1),   u(0) = u(1) = 0,   u*(x) = sin(3 pi x).

The boundary condition is imposed as a hard constraint, ``u_theta(x) = x (1 - x)
N_theta(x)``, so the objective is the PDE residual alone. That keeps the loss
single-term, which is deliberately unlike the multi-term PINN losses of Section
2.1; the point here is a runnable demo of the optimizer, not a benchmark.

Run with ``python -m ss_esoap.train``.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .optimizer import soap, ss_esoap

LAYERS = (1, 32, 32, 1)
N_COLLOCATION = 256
STEPS = 400


def init_params(key, layers=LAYERS):
    params = {}
    for i, (a, b) in enumerate(zip(layers[:-1], layers[1:])):
        key, sub = jax.random.split(key)
        # Glorot, which is what tanh MLPs in the PINN literature normally use.
        scale = np.sqrt(2.0 / (a + b))
        params[f"w{i}"] = scale * jax.random.normal(sub, (a, b))
        params[f"b{i}"] = jnp.zeros((b,))
    return params


def net(params, x):
    h = jnp.atleast_1d(x)
    n = len(LAYERS) - 1
    for i in range(n):
        h = h @ params[f"w{i}"] + params[f"b{i}"]
        if i < n - 1:
            h = jnp.tanh(h)
    return h[0]


def u(params, x):
    """Hard-constrained trial solution; ``u(0) = u(1) = 0`` holds by construction."""
    return x * (1.0 - x) * net(params, x)


def source(x):
    return 9.0 * jnp.pi**2 * jnp.sin(3.0 * jnp.pi * x)


def exact(x):
    return jnp.sin(3.0 * jnp.pi * x)


def residual(params, x):
    uxx = jax.grad(jax.grad(u, argnums=1), argnums=1)(params, x)
    return -uxx - source(x)


def loss_fn(params, xs):
    return jnp.mean(jax.vmap(residual, (None, 0))(params, xs) ** 2)


def l2_error(params, xs):
    pred = jax.vmap(u, (None, 0))(params, xs)
    ref = exact(xs)
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def run(opt, params, xs, steps=STEPS):
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params, xs)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    for _ in range(steps):
        params, state, loss = step(params, state)
        losses.append(float(loss))
    return params, losses


LRS = (3e-3, 1e-2, 3e-2)


def main():
    """Each optimizer gets the same small learning-rate sweep and reports its
    best run, which is the tuning protocol of Appendix C.4 in miniature. Reading
    one hand-picked rate off each would say more about the pick than the method.
    """
    xs = jnp.linspace(0.0, 1.0, N_COLLOCATION)
    grid = jnp.linspace(0.0, 1.0, 1001)
    p0 = init_params(jax.random.key(0))

    makes = {
        "adam": optax.adam,
        "soap": lambda lr: soap(lr, precondition_frequency=50),
        "ss-esoap": ss_esoap,
    }
    print(f"1D Poisson PINN, {LAYERS}, {N_COLLOCATION} points, {STEPS} steps")
    print(f"best of eta in {LRS}")
    print(f"{'optimizer':<10} {'eta':>7} {'residual':>12} {'rel L2':>10} {'sec':>7}")
    for name, make in makes.items():
        t0 = time.perf_counter()
        best = None
        for lr in LRS:
            params, losses = run(make(lr), p0, xs)
            if best is None or losses[-1] < best[1]:
                best = (lr, losses[-1], float(l2_error(params, grid)))
        dt = time.perf_counter() - t0
        print(
            f"{name:<10} {best[0]:>7.0e} {best[1]:>12.4e} {best[2]:>10.4e} {dt:>7.2f}"
        )


if __name__ == "__main__":
    main()
