"""Train only the linear readout c; omega and b stay frozen."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def adam_init(c: Array) -> dict:
    return {
        "m": jnp.zeros_like(c),
        "v": jnp.zeros_like(c),
        "t": jnp.array(0, dtype=jnp.int32),
    }


def adam_update(
    c: Array,
    g: Array,
    state: dict,
    lr: float = 1e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[Array, dict]:
    t = state["t"] + 1
    m = b1 * state["m"] + (1.0 - b1) * g
    v = b2 * state["v"] + (1.0 - b2) * (g ** 2)
    mhat = m / (1.0 - b1 ** t)
    vhat = v / (1.0 - b2 ** t)
    c_new = c - lr * mhat / (jnp.sqrt(vhat) + eps)
    return c_new, {"m": m, "v": v, "t": t}


def train_step(
    c: Array,
    frozen: dict,
    loss_fn: Callable[[Array], Array],
    opt_state: dict,
    lr: float = 1e-2,
) -> Tuple[Array, dict, dict, Array]:
    """One Adam step on c. `frozen` is returned unchanged."""
    loss, g = jax.value_and_grad(loss_fn)(c)
    c_new, opt_state = adam_update(c, g, opt_state, lr=lr)
    return c_new, frozen, opt_state, loss


def train_readout(
    c: Array,
    loss_fn: Callable[[Array], Array],
    n_steps: int,
    lr: float = 1e-2,
    jit: bool = True,
) -> Tuple[Array, Array]:
    """Adam on c for n_steps. Returns (c, last_loss)."""
    state = adam_init(c)

    def step(carry, _):
        c_i, st = carry
        loss, g = jax.value_and_grad(loss_fn)(c_i)
        c_i, st = adam_update(c_i, g, st, lr=lr)
        return (c_i, st), loss

    if jit:
        step = jax.jit(step)
    (c, state), losses = jax.lax.scan(step, (c, state), xs=None, length=n_steps)
    return c, losses[-1]


def bfgs_refine(
    c: Array,
    loss_fn: Callable[[Array], Array],
    maxiter: int = 50,
) -> Array:
    """Optional BFGS polish (paper uses L-BFGS after Adam)."""
    from jax.scipy.optimize import minimize

    result = minimize(loss_fn, c, method="BFGS", options={"maxiter": maxiter})
    return result.x


def ridge_readout(
    linear_maps: Sequence[Tuple[Array, Array]],
    lam_l2: float = 1e-6,
) -> Array:
    """Min-norm linear readout for linear PDEs: mean ||A c - y||^2 + lam ||c||^2.

    The paper notes that linear constraints reduce to c = A^+ y.
    """
    n = linear_maps[0][0].shape[1]
    h = lam_l2 * jnp.eye(n)
    rhs = jnp.zeros((n,))
    for a_mat, y in linear_maps:
        m = a_mat.shape[0]
        h = h + (a_mat.T @ a_mat) / m
        rhs = rhs + (a_mat.T @ y) / m
    return jnp.linalg.solve(h, rhs)


def jacobian_map(fn: Callable[[Array], Array], n: int) -> Array:
    """Jacobian of a map c |-> residual that is linear in c."""
    return jax.jacfwd(fn)(jnp.zeros((n,)))
