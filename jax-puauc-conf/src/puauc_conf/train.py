"""SGD on the eq. (17) training objective (Algorithm 1, CPU-friendly)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from puauc_conf.model import Params, apply_score, init_linear, sgd_step
from puauc_conf.risk import U_CLIP, roc_auc, training_puauc_loss


def logistic_nll(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Binary logistic loss for û(x) ≈ p(o=1|x)."""
    return jnp.mean(jax.nn.softplus(logits) - labels * logits)


def fit_labeling_classifier(
    x_labeled: jax.Array,
    x_unlabeled: jax.Array,
    steps: int = 200,
    lr: float = 0.2,
    key: jax.Array | None = None,
) -> Params:
    """Train a linear û(x) on labeled vs unlabeled (Algorithm 1 line 2)."""
    x = jnp.concatenate([x_labeled, x_unlabeled], axis=0)
    y = jnp.concatenate(
        [jnp.ones((x_labeled.shape[0],)), jnp.zeros((x_unlabeled.shape[0],))],
        axis=0,
    )
    if key is None:
        key = jax.random.PRNGKey(0)
    params = init_linear(key, x.shape[-1], scale=0.05)

    def loss_fn(p):
        return logistic_nll(apply_score(p, x), y)

    grad_fn = jax.jit(jax.grad(loss_fn))
    for _ in range(steps):
        params = sgd_step(params, grad_fn(params), lr)
    return params


def predict_u(u_params: Params, x: jax.Array, clip: float = U_CLIP) -> jax.Array:
    p = jax.nn.sigmoid(apply_score(u_params, x))
    return jnp.maximum(p, clip)


def train_score(
    x_labeled: jax.Array,
    r: jax.Array,
    x_unlabeled: jax.Array,
    u_hat_labeled: jax.Array,
    params: Params,
    steps: int = 300,
    lr: float = 0.05,
    batch_p: int | None = None,
    batch_u: int | None = None,
    key: jax.Array | None = None,
) -> tuple[Params, list[float]]:
    """Minimize the training form of eq. (17) with plain SGD (Algorithm 1)."""
    n_p = int(x_labeled.shape[0])
    n_u = int(x_unlabeled.shape[0])
    alpha_hat = n_p / (n_p + n_u)
    if batch_p is None:
        batch_p = n_p
    if batch_u is None:
        batch_u = n_u
    batch_p = min(batch_p, n_p)
    batch_u = min(batch_u, n_u)
    if key is None:
        key = jax.random.PRNGKey(1)

    def loss_fn(p, xp, ru, uh, xu):
        s_p = apply_score(p, xp)
        s_u = apply_score(p, xu)
        return training_puauc_loss(s_p, s_u, ru, uh, alpha_hat=alpha_hat)

    grad_fn = jax.jit(jax.grad(loss_fn))
    full_loss = jax.jit(
        lambda p: loss_fn(p, x_labeled, r, u_hat_labeled, x_unlabeled)
    )
    start = float(full_loss(params))
    for _ in range(steps):
        key, kp, ku = jax.random.split(key, 3)
        ip = jax.random.choice(kp, n_p, (batch_p,), replace=False)
        iu = jax.random.choice(ku, n_u, (batch_u,), replace=False)
        g = grad_fn(
            params, x_labeled[ip], r[ip], u_hat_labeled[ip], x_unlabeled[iu]
        )
        params = sgd_step(params, g, lr)
    end = float(full_loss(params))
    return params, [start, end]


def evaluate_auc(params: Params, x: jax.Array, y: jax.Array) -> float:
    scores = apply_score(params, x)
    return float(roc_auc(scores, y))
