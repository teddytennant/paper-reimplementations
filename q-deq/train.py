"""Train a shared-layer Q-DEQ on a synthetic multivariate series (CPU)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from q_deq._host import preload

preload()

import jax
import jax.numpy as jnp
import numpy as np

from q_deq.deq import init_head, init_shared_params
from q_deq.solver import training_forward

LOOKBACK = 8
N_VARIATES = 3
HORIZON = 1
HIDDEN = 16
INPUT_DIM = LOOKBACK * N_VARIATES
OUT_DIM = HORIZON * N_VARIATES
STEPS = 200
BATCH = 16
LR = 5e-2


def make_var_dataset(n_seq=64, length=16, n_variates=3, seed=0):
    """Synthetic VAR(1) windows; no public MTS datasets."""
    rng = np.random.default_rng(seed)
    A = 0.6 * np.eye(n_variates) + 0.15 * rng.normal(size=(n_variates, n_variates))
    eig = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (0.7 / max(eig, 1e-6))
    xs, ys = [], []
    for _ in range(n_seq):
        s = rng.normal(size=(n_variates,))
        traj = [s]
        for _t in range(length - 1):
            s = A @ s + 0.1 * rng.normal(size=(n_variates,))
            traj.append(s)
        traj = np.stack(traj, axis=0)
        window = traj[:LOOKBACK].reshape(-1)
        target = traj[LOOKBACK : LOOKBACK + HORIZON].reshape(-1)
        xs.append(window)
        ys.append(target)
    return jnp.asarray(np.stack(xs), dtype=jnp.float32), jnp.asarray(
        np.stack(ys), dtype=jnp.float32
    )


def init_adam(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


def adam_update(params, grads, state, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1

    def upd(p, g, m, v):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g ** 2)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        p = p - lr * mhat / (jnp.sqrt(vhat) + eps)
        return p, m, v

    leaves_p, treedef = jax.tree_util.tree_flatten(params)
    leaves_g = jax.tree_util.tree_leaves(grads)
    leaves_m = jax.tree_util.tree_leaves(state["m"])
    leaves_v = jax.tree_util.tree_leaves(state["v"])
    new_p, new_m, new_v = [], [], []
    for p, g, m, v in zip(leaves_p, leaves_g, leaves_m, leaves_v):
        p, m, v = upd(p, g, m, v)
        new_p.append(p)
        new_m.append(m)
        new_v.append(v)
    params = jax.tree_util.tree_unflatten(treedef, new_p)
    state = {
        "m": jax.tree_util.tree_unflatten(treedef, new_m),
        "v": jax.tree_util.tree_unflatten(treedef, new_v),
        "t": t,
    }
    return params, state


def main():
    key = jax.random.PRNGKey(0)
    X, Y = make_var_dataset()
    n = X.shape[0]
    k1, k2, key = jax.random.split(key, 3)
    layer = init_shared_params(k1, hidden=HIDDEN, input_dim=INPUT_DIM, mlp_hidden=16)
    hd = init_head(k2, HIDDEN, OUT_DIM)
    params = {"layer": layer, "head": hd}
    opt = init_adam(params)

    def loss_on_batch(params, xb, yb, keys):
        def one(x, y, k):
            pred = training_forward(
                params["layer"],
                params["head"],
                x,
                k,
                k_inner=4,
                n_dirs=2,
                n_sweeps=30,
                n_restarts=1,
                use_w8a8=True,
            )
            return jnp.mean((pred - y) ** 2)

        return jnp.mean(jax.vmap(one)(xb, yb, keys))

    @jax.jit
    def step(params, opt, xb, yb, keys):
        loss, grads = jax.value_and_grad(loss_on_batch)(params, xb, yb, keys)
        params, opt = adam_update(params, grads, opt)
        return params, opt, loss

    def batch_at(params_key, i):
        idx = jax.random.permutation(params_key, n)[:BATCH]
        return X[idx], Y[idx]

    # Initial MSE on a fixed eval batch.
    eval_x, eval_y = X[:BATCH], Y[:BATCH]
    k_eval = jax.random.split(jax.random.PRNGKey(123), BATCH)
    init_mse = float(loss_on_batch(params, eval_x, eval_y, k_eval))
    print(f"initial MSE: {init_mse:.6f}")

    for i in range(STEPS):
        key, kb, ks = jax.random.split(key, 3)
        xb, yb = batch_at(kb, i)
        bkeys = jax.random.split(ks, BATCH)
        params, opt, loss = step(params, opt, xb, yb, bkeys)
        if i == 0 or (i + 1) % 50 == 0:
            print(f"step {i+1:3d} train MSE: {float(loss):.6f}")

    final_mse = float(loss_on_batch(params, eval_x, eval_y, k_eval))
    print(f"final MSE: {final_mse:.6f}")
    if not (final_mse < init_mse):
        raise SystemExit(f"loss did not drop ({init_mse} -> {final_mse})")


if __name__ == "__main__":
    main()
