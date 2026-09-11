"""Confidence-weighted PU-AUC risk (arXiv:2609.10928).

Equations follow the paper:
  (2)  zero-one AUC risk
  (3)/(9)  sigmoid-smoothed AUC risk
  (4)  empirical PN estimator
  (12) rewritten risk using p and p^p (Lemma C.1)
  (15) rewritten risk using labeled density and confidence
  (16) split of (15) into labeled/unlabeled densities
  (17) empirical estimator (omit n=m in the labeled-labeled term)
  (18) / Prop. 4.1  strictly increasing transform of the posterior
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Paper §E.3: round up û(x^p) below 0.01 to 0.01.
U_CLIP = 0.01


def sigmoid_surrogate(z: jax.Array) -> jax.Array:
    """σ(z) = 1 / (1 + exp(-z)), the paper's pairwise surrogate."""
    return jax.nn.sigmoid(z)


def pairwise_sigmoid(s_left: jax.Array, s_right: jax.Array) -> jax.Array:
    """f(x, x') = σ(-s(x) + s(x')) as in eq. (9). Shape (len(left), len(right))."""
    s_left = jnp.asarray(s_left).reshape(-1)
    s_right = jnp.asarray(s_right).reshape(-1)
    return sigmoid_surrogate(-s_left[:, None] + s_right[None, :])


def zero_one_pairwise(s_pos: jax.Array, s_neg: jax.Array) -> jax.Array:
    """ℓ_01(s(x^p) - s(x^n)) from eq. (2): 1 if z<0, 0 if z>0, 1/2 if z=0."""
    z = jnp.asarray(s_pos).reshape(-1)[:, None] - jnp.asarray(s_neg).reshape(-1)[None, :]
    return jnp.where(z < 0.0, 1.0, jnp.where(z > 0.0, 0.0, 0.5))


def pn_auc_risk(s_pos: jax.Array, s_neg: jax.Array) -> jax.Array:
    """Empirical PN sigmoid AUC risk, eq. (4)."""
    return pairwise_sigmoid(s_pos, s_neg).mean()


def pn_zero_one_auc_risk(s_pos: jax.Array, s_neg: jax.Array) -> jax.Array:
    """Empirical zero-one AUC risk, eq. (2)."""
    return zero_one_pairwise(s_pos, s_neg).mean()


def roc_auc(scores: jax.Array, y: jax.Array) -> jax.Array:
    """AUC = 1 - R_01 on a labeled sample (ties count as 1/2)."""
    scores = jnp.asarray(scores).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    s_pos = scores[y == 1]
    s_neg = scores[y == 0]
    return 1.0 - pn_zero_one_auc_risk(s_pos, s_neg)


def population_auc_risk_pn(scores: jax.Array, y: jax.Array) -> jax.Array:
    """Finite-population version of eq. (9): mean pairwise surrogate over PN pairs."""
    scores = jnp.asarray(scores).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    return pn_auc_risk(scores[y == 1], scores[y == 0])


def population_auc_risk_eq12(scores: jax.Array, y: jax.Array) -> jax.Array:
    """Eq. (12): (1/(1-π)) E_{p^p} E_p [f] - π / (2(1-π))."""
    scores = jnp.asarray(scores).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    pi = y.mean()
    s_pos = scores[y == 1]
    term = pairwise_sigmoid(s_pos, scores).mean()
    return term / (1.0 - pi) - pi / (2.0 * (1.0 - pi))


def population_auc_risk_rewritten(
    s_labeled: jax.Array,
    s_all: jax.Array,
    r: jax.Array,
    u: jax.Array,
    c: jax.Array | float,
    pi: jax.Array | float,
) -> jax.Array:
    """Eq. (15) on a finite population (true r = η, true u = p(o=1|x))."""
    w = jnp.asarray(r).reshape(-1) / jnp.asarray(u).reshape(-1)
    f = pairwise_sigmoid(s_labeled, s_all)
    term = (w[:, None] * f).mean()
    return (c / (1.0 - pi)) * term - pi / (2.0 * (1.0 - pi))


def population_auc_risk_split(
    s_labeled: jax.Array,
    s_unlabeled: jax.Array,
    r: jax.Array,
    u: jax.Array,
    alpha: jax.Array | float,
    c: jax.Array | float,
    pi: jax.Array | float,
) -> jax.Array:
    """Eq. (16): split of eq. (15) into labeled and unlabeled densities."""
    w = jnp.asarray(r).reshape(-1) / jnp.asarray(u).reshape(-1)
    f_ll = pairwise_sigmoid(s_labeled, s_labeled)
    f_lu = pairwise_sigmoid(s_labeled, s_unlabeled)
    term_ll = (w[:, None] * f_ll).mean()
    term_lu = (w[:, None] * f_lu).mean()
    inner = alpha * term_ll + (1.0 - alpha) * term_lu
    return (c / (1.0 - pi)) * inner - pi / (2.0 * (1.0 - pi))


def eq17_pairwise_weights(
    r: jax.Array,
    u_hat: jax.Array,
    n_unlabeled: int,
    alpha_hat: jax.Array | float | None = None,
    c: jax.Array | float | None = None,
    pi: jax.Array | float | None = None,
    u_clip: float = U_CLIP,
) -> tuple[jax.Array, jax.Array]:
    """Pairwise weights in eq. (17).

    Returns
    -------
    w_ll : (N^p, N^p)
        Weight on f(x_n^p, x̄_m^p); diagonal is zero (n ≠ m).
    w_lu : (N^p, N)
        Weight on f(x_n^p, x_m).
    """
    r = jnp.asarray(r).reshape(-1)
    u_hat = jnp.maximum(jnp.asarray(u_hat).reshape(-1), u_clip)
    n_p = r.shape[0]
    n_u = int(n_unlabeled)
    if alpha_hat is None:
        alpha_hat = n_p / (n_p + n_u)
    ratio = r / u_hat
    w_ll = (alpha_hat * ratio[:, None]) / (n_p * (n_p - 1))
    w_ll = w_ll * (1.0 - jnp.eye(n_p, dtype=w_ll.dtype))
    w_lu = ((1.0 - alpha_hat) * ratio[:, None]) / (n_p * n_u)
    if c is not None and pi is not None:
        scale = c / (1.0 - pi)
        w_ll = scale * w_ll
        w_lu = scale * w_lu
    return w_ll, w_lu


def empirical_puauc_risk(
    s_labeled: jax.Array,
    s_unlabeled: jax.Array,
    r: jax.Array,
    u_hat: jax.Array,
    alpha_hat: jax.Array | float | None = None,
    c: jax.Array | float | None = None,
    pi: jax.Array | float | None = None,
    u_clip: float = U_CLIP,
) -> jax.Array:
    """Empirical estimator eq. (17).

    If ``c`` or ``pi`` is omitted, drop the positive factor c/(1-π) and the
    additive constant π/(2(1-π)), which the paper notes do not affect
    optimization of s.
    """
    s_labeled = jnp.asarray(s_labeled).reshape(-1)
    s_unlabeled = jnp.asarray(s_unlabeled).reshape(-1)
    r = jnp.asarray(r).reshape(-1)
    u_hat = jnp.maximum(jnp.asarray(u_hat).reshape(-1), u_clip)
    n_p = s_labeled.shape[0]
    n_u = s_unlabeled.shape[0]
    if alpha_hat is None:
        alpha_hat = n_p / (n_p + n_u)
    w_ll, w_lu = eq17_pairwise_weights(
        r, u_hat, n_u, alpha_hat=alpha_hat, c=None, pi=None, u_clip=u_clip
    )
    inner = (w_ll * pairwise_sigmoid(s_labeled, s_labeled)).sum() + (
        w_lu * pairwise_sigmoid(s_labeled, s_unlabeled)
    ).sum()
    if c is None or pi is None:
        return inner
    return (c / (1.0 - pi)) * inner - pi / (2.0 * (1.0 - pi))


def training_puauc_loss(
    s_labeled: jax.Array,
    s_unlabeled: jax.Array,
    r: jax.Array,
    u_hat: jax.Array,
    alpha_hat: jax.Array | float | None = None,
    u_clip: float = U_CLIP,
) -> jax.Array:
    """Training objective: eq. (17) without c/(1-π) and π/(2(1-π))."""
    return empirical_puauc_risk(
        s_labeled,
        s_unlabeled,
        r,
        u_hat,
        alpha_hat=alpha_hat,
        c=None,
        pi=None,
        u_clip=u_clip,
    )
