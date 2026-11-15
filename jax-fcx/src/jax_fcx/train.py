"""Algorithm 1: train the conditional VAE for FCx, classifier frozen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_fcx.classifier import predict as classifier_predict
from jax_fcx.classifier import train_classifier
from jax_fcx.constraints import unary_penalty
from jax_fcx.losses import fcx_objective
from jax_fcx.metrics import feasibility_rate, proximity, sparsity_score, validity_rate
from jax_fcx.vae import generate_cf, init_cvae


def _adam_init(params: Any) -> dict[str, Any]:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.array(0, dtype=jnp.int32)}


def _adam_update(
    params: Any,
    grads: Any,
    state: dict[str, Any],
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    t = state["t"] + 1
    m = jax.tree.map(lambda mm, g: b1 * mm + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda vv, g: b2 * vv + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda mm: mm / (1.0 - b1 ** t), m)
    vhat = jax.tree.map(lambda vv: vv / (1.0 - b2 ** t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


@dataclass
class FCxConfig:
    d: int = 8
    z_dim: int = 8
    hidden: int = 32
    n_linear: int = 5
    n_classes: int = 2
    lr: float = 1e-2
    steps: int = 40
    batch_size: int = 32
    binary: bool = False
    margin: float = 1.0
    k_lof: int = 5
    age_idx: int = 0
    edu_idx: int = 1
    immutable_idx: int = 2
    alpha: float = 0.0
    beta: float = 1.0
    theta: float = 0.1
    lambda_rec: float = 0.5
    lambda_val: float = 2.0
    lambda_feas: float = 2.0
    lambda_causal: float = 0.1
    lambda_spar: float = 0.5
    lambda_lof: float = 0.1
    lambda_nc: float = 1.0
    lambda_c: float = 1.0
    source_class: int | None = 0
    adjacency: np.ndarray | None = field(default=None, repr=False)


def make_synthetic_data(
    n: int = 512,
    d: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Tabular synthetic data: age, education, immutable race dummy, rest mutable.

    Labels are income-like: 1 if age + edu + noise exceeds a threshold.
    Hard constraints: age_cf >= age; if edu_cf > edu then age_cf > age.
    A encodes a known age -> education edge (no PC-algorithm).
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.0, 1.0, size=(n, 1))
    edu = np.clip(0.65 * age + 0.35 * rng.uniform(0.0, 1.0, size=(n, 1)), 0.0, 1.0)
    race = rng.uniform(0.0, 1.0, size=(n, 1))
    rest = rng.uniform(0.0, 1.0, size=(n, d - 3))
    x = np.concatenate([age, edu, race, rest], axis=1).astype(np.float32)
    noise = rng.normal(0.0, 0.04, size=n).astype(np.float32)
    y = ((x[:, 0] + x[:, 1] + 0.1 * x[:, 3] + noise) > 1.15).astype(np.int32)
    A = np.zeros((d, d), dtype=np.float32)
    A[0, 1] = 1.0
    mutable_idx = np.array([i for i in range(d) if i != 2], dtype=np.int32)
    immutable_idx = np.array([2], dtype=np.int32)
    return {
        "x": x,
        "y": y,
        "A": A,
        "mutable_idx": mutable_idx,
        "immutable_idx": immutable_idx,
        "age_idx": 0,
        "edu_idx": 1,
    }


def train_fcx(
    key: jax.Array,
    x: jax.Array,
    clf_params: dict,
    config: FCxConfig | None = None,
    adjacency: jax.Array | None = None,
    mutable_idx: jax.Array | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Run Algorithm 1. Classifier params stay frozen. Returns (vae_params, logs)."""
    cfg = config if config is not None else FCxConfig()
    x = jnp.asarray(x)
    n, d = x.shape
    if mutable_idx is None:
        mutable_idx = jnp.array([i for i in range(d) if i != cfg.immutable_idx], dtype=jnp.int32)
    else:
        mutable_idx = jnp.asarray(mutable_idx)
    if adjacency is None:
        if cfg.adjacency is not None:
            adjacency = jnp.asarray(cfg.adjacency)
        else:
            adjacency = jnp.zeros((d, d), dtype=x.dtype)
            adjacency = adjacency.at[cfg.age_idx, cfg.edu_idx].set(1.0)
    else:
        adjacency = jnp.asarray(adjacency)

    pred = classifier_predict(clf_params, x)
    y_target = (1 - pred).astype(jnp.int32)
    if cfg.source_class is not None:
        mask = pred == cfg.source_class
        if int(jnp.sum(mask)) >= 2:
            x = x[mask]
            y_target = y_target[mask]
            n = x.shape[0]

    d_mutable = int(mutable_idx.shape[0])
    z_dim = cfg.z_dim if cfg.z_dim is not None else d_mutable
    key, k_init = jax.random.split(key)
    vae_params = init_cvae(
        k_init,
        d_mutable=d_mutable,
        z_dim=z_dim,
        hidden=cfg.hidden,
        n_linear=cfg.n_linear,
        n_classes=cfg.n_classes,
    )
    opt = _adam_init(vae_params)
    bs = min(cfg.batch_size, n)
    k_lof = min(cfg.k_lof, max(bs - 1, 1))

    def loss_fn(p, bx, by, rng):
        return fcx_objective(
            p,
            clf_params,
            bx,
            by,
            rng,
            mutable_idx,
            adjacency,
            binary=cfg.binary,
            margin=cfg.margin,
            k_lof=k_lof,
            age_idx=cfg.age_idx,
            edu_idx=cfg.edu_idx,
            alpha=cfg.alpha,
            beta=cfg.beta,
            theta=cfg.theta,
            lambda_rec=cfg.lambda_rec,
            lambda_val=cfg.lambda_val,
            lambda_feas=cfg.lambda_feas,
            lambda_causal=cfg.lambda_causal,
            lambda_spar=cfg.lambda_spar,
            lambda_lof=cfg.lambda_lof,
            lambda_nc=cfg.lambda_nc,
            lambda_c=cfg.lambda_c,
        )

    @jax.jit
    def step(p, opt_state, bx, by, rng):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, bx, by, rng)
        p, opt_state = _adam_update(p, grads, opt_state, lr=cfg.lr)
        return p, opt_state, loss, parts

    # fixed prefix batch for step-0 vs final metrics
    bx0, by0 = x[:bs], y_target[:bs]
    key, k0 = jax.random.split(key)
    loss0, parts0 = loss_fn(vae_params, bx0, by0, k0)
    x_cf0 = generate_cf(vae_params, bx0, by0, mutable_idx, key=None)
    pred_cf0 = classifier_predict(clf_params, x_cf0)

    losses: list[float] = [float(loss0)]
    feas_viol: list[float] = [float(parts0["feas"])]
    val_rates: list[float] = [float(validity_rate(pred_cf0, by0))]
    feas_rates: list[float] = [float(feasibility_rate(bx0, x_cf0, binary=cfg.binary))]

    for _ in range(cfg.steps):
        key, k_perm, k_step = jax.random.split(key, 3)
        idx = jax.random.permutation(k_perm, n)[:bs]
        vae_params, opt, loss, parts = step(vae_params, opt, x[idx], y_target[idx], k_step)
        losses.append(float(loss))
        feas_viol.append(float(parts["feas"]))

    x_cfN = generate_cf(vae_params, bx0, by0, mutable_idx, key=None)
    pred_cfN = classifier_predict(clf_params, x_cfN)
    val_rates.append(float(validity_rate(pred_cfN, by0)))
    feas_rates.append(float(feasibility_rate(bx0, x_cfN, binary=cfg.binary)))
    feas_viol.append(float(jnp.mean(unary_penalty(bx0, x_cfN, age_idx=cfg.age_idx))))

    logs = {
        "loss": losses,
        "feas_violation": feas_viol,
        "validity": val_rates,
        "feasibility_rate": feas_rates,
        "x": x,
        "y_target": y_target,
        "mutable_idx": mutable_idx,
        "adjacency": adjacency,
        "eval_x": bx0,
        "eval_y": by0,
    }
    return vae_params, logs


def evaluate_cf(
    vae_params: dict,
    clf_params: dict,
    x: jax.Array,
    y_target: jax.Array,
    mutable_idx: jax.Array,
    binary: bool = False,
) -> dict[str, float]:
    x_cf = generate_cf(vae_params, x, y_target, mutable_idx, key=None)
    pred_cf = classifier_predict(clf_params, x_cf)
    return {
        "validity": float(validity_rate(pred_cf, y_target)),
        "feasibility": float(feasibility_rate(x, x_cf, binary=binary)),
        "sparsity": float(sparsity_score(x, x_cf)),
        "proximity": float(proximity(x, x_cf)),
        "unary_violation": float(jnp.mean(unary_penalty(x, x_cf))),
    }


def train_classifier_then_fcx(
    key: jax.Array,
    n: int = 512,
    clf_steps: int = 200,
    config: FCxConfig | None = None,
) -> tuple[dict, dict, dict[str, Any], dict[str, Any]]:
    """Convenience: synthetic data, train classifier, freeze, run Algorithm 1."""
    cfg = config if config is not None else FCxConfig()
    data = make_synthetic_data(n=n, d=cfg.d, seed=0)
    x = jnp.asarray(data["x"])
    y = jnp.asarray(data["y"])
    key, k_clf, k_fcx = jax.random.split(key, 3)
    clf_params, _ = train_classifier(k_clf, x, y, hidden=cfg.hidden, steps=clf_steps)
    vae_params, logs = train_fcx(
        k_fcx,
        x,
        clf_params,
        config=cfg,
        adjacency=jnp.asarray(data["A"]),
        mutable_idx=jnp.asarray(data["mutable_idx"]),
    )
    return clf_params, vae_params, logs, data
