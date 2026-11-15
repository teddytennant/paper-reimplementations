"""Algorithm 1: self-bounding optimization of the Dirichlet hyper-posterior."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_dis_smv.bounds import dis_r_certificate, dis_r_penalty, dis_v_certificate, dis_v_penalty, kl_inv
from jax_dis_smv.majority_vote import empirical_risk_01, empirical_risk_sigmoid


class TrainResult(NamedTuple):
    alpha: jax.Array
    rho: jax.Array
    rhat: jax.Array
    bound: jax.Array


def _inv_softplus(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(jnp.clip(x, 1.0e-6, 20.0))))


def _min_concentration(beta: jax.Array, objective: str, lam: float) -> jax.Array:
    floor = jnp.full_like(beta, 1.0e-3)
    if objective == "dis-v":
        # Eq 12 is finite iff alpha_j > ((lam-1)/lam) beta_j.
        floor = jnp.maximum(floor, ((lam - 1.0) / lam) * beta + 1.0e-3)
    return floor


def split_data(X: jax.Array, y: jax.Array, n1: int, key: jax.Array | None = None):
    """Split S into disjoint S1, S2 with |S1|=n1 and |S1|+|S2|=n (Section 4.2)."""
    n = int(X.shape[0])
    n1 = int(n1)
    if n1 < 0 or n1 > n:
        raise ValueError("n1 must satisfy 0 <= n1 <= n")
    idx = jnp.arange(n)
    if key is not None:
        idx = jax.random.permutation(key, n)
    i1, i2 = idx[:n1], idx[n1:]
    return (X[i1], y[i1]), (X[i2], y[i2])


def synthetic_binary_2d(key: jax.Array, n: int = 64) -> tuple[jax.Array, jax.Array]:
    """Linearly separable 2D binary labels: y = I[x_1 > 0]."""
    X = jax.random.uniform(key, (n, 2), minval=-1.0, maxval=1.0)
    y = (X[:, 0] > 0.0).astype(jnp.int32)
    return X, y


def axis_aligned_stumps(X: jax.Array, thresholds_per_dim: int = 4) -> jax.Array:
    """Handful of axis-aligned stump voters: x_d > t and x_d <= t."""
    X = jnp.asarray(X)
    n, d = X.shape
    cols = []
    for dim in range(int(d)):
        lo = jnp.min(X[:, dim])
        hi = jnp.max(X[:, dim])
        # interior thresholds, plus 0 when it lies in range (perfect stump for x1>0)
        ts = jnp.linspace(lo, hi, thresholds_per_dim + 2)[1:-1]
        ts = jnp.unique(jnp.concatenate([ts, jnp.array([0.0], dtype=X.dtype)]))
        for t in ts.tolist():
            cols.append((X[:, dim] > t).astype(jnp.int32))
            cols.append((X[:, dim] <= t).astype(jnp.int32))
    return jnp.stack(cols, axis=1)


def algorithm_1(
    key: jax.Array,
    voter_preds: jax.Array,
    y: jax.Array,
    beta: jax.Array,
    *,
    delta: float = 0.05,
    n_epochs: int = 20,
    eta: float = 0.1,
    batch_size: int | None = None,
    objective: str = "dis-r",
    lam: float = 1.5,
    slope: float = 100.0,
    n_classes: int = 2,
    alpha_init: jax.Array | None = None,
) -> TrainResult:
    """Self-bounding Dirichlet hyper-posterior (Algorithm 1).

    Each mini-batch draws rho ~ Dir(alpha), builds MVrho, evaluates
    B(alpha; U, P, delta), and takes a gradient step on unconstrained
    parameters whose softplus image stays in R>0. After T epochs a final
    rho is drawn. Training uses the sigmoid surrogate of Section 6; the
    returned certificate uses empirical 0-1 loss on the full sample.
    """
    if objective not in ("dis-r", "dis-v"):
        raise ValueError("objective must be 'dis-r' or 'dis-v'")

    voter_preds = jnp.asarray(voter_preds)
    y = jnp.asarray(y)
    beta = jnp.asarray(beta)
    n, m = int(voter_preds.shape[0]), int(voter_preds.shape[1])
    if batch_size is None or batch_size >= n:
        batch_size = n

    key, k_init = jax.random.split(key)
    if alpha_init is None:
        # Algorithm 1 requires alpha > 0; ones is a stable simplex-centered start
        # (the paper's U[0.01, 2] experimental draw is too sparse for tiny ensembles).
        alpha_init = jnp.ones((m,), dtype=beta.dtype)
    else:
        alpha_init = jnp.asarray(alpha_init)

    min_conc = _min_concentration(beta, objective, lam)
    alpha_init = jnp.maximum(alpha_init, min_conc + 1.0e-3)
    theta = _inv_softplus(alpha_init - min_conc)

    def alpha_of(th):
        return min_conc + jax.nn.softplus(th)

    def batch_loss(th, rho_key, idx):
        alpha = alpha_of(th)
        rho = jax.random.dirichlet(rho_key, alpha)
        vp = voter_preds[idx]
        yb = y[idx]
        rhat = empirical_risk_sigmoid(vp, rho, yb, slope=slope, n_classes=n_classes)
        n_b = idx.shape[0]
        if objective == "dis-r":
            eps = dis_r_penalty(alpha, beta, rho, n_b, delta)
        else:
            eps = dis_v_penalty(alpha, beta, n_b, delta, lam)
        # Inverse-kl saturates when the penalty is vacuous; adding the sigmoid
        # risk keeps a nonzero training signal (kl_inv is increasing in rhat).
        return kl_inv(rhat, eps) + rhat

    @jax.jit
    def step(th, rho_key, idx):
        loss, grads = jax.value_and_grad(batch_loss)(th, rho_key, idx)
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=1.0, neginf=-1.0)
        return th - eta * grads, loss

    n_batches = (n + batch_size - 1) // batch_size
    for _ in range(int(n_epochs)):
        key, k_perm = jax.random.split(key)
        order = jax.random.permutation(k_perm, n) if batch_size < n else jnp.arange(n)
        for b in range(n_batches):
            idx = order[b * batch_size : min((b + 1) * batch_size, n)]
            key, k_rho = jax.random.split(key)
            theta, _ = step(theta, k_rho, idx)

    alpha = alpha_of(theta)
    key, k_final = jax.random.split(key)
    rho = jax.random.dirichlet(k_final, alpha)
    rhat = empirical_risk_01(voter_preds, rho, y, n_classes=n_classes)
    if objective == "dis-r":
        bound = dis_r_certificate(rhat, alpha, beta, rho, n, delta)
    else:
        bound = dis_v_certificate(rhat, alpha, beta, n, delta, lam)
    return TrainResult(alpha=alpha, rho=rho, rhat=rhat, bound=bound)
