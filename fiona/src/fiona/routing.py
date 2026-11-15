"""Ternary candidates, group routing loss, sensitivity, and protection."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

KAPPA_DEFAULT = 8.0


class TernaryCandidates(NamedTuple):
    q: jnp.ndarray
    gamma: jnp.ndarray
    W_hat: jnp.ndarray


def ternary_quantize(W: jnp.ndarray) -> TernaryCandidates:
    """Per-output mean-abs scale and clip-round ternary candidates.

    gamma_o = mean_i |W[o, i]|. If gamma_o == 0, q[o] = 0.
    Otherwise q[o, i] = clip(round(W[o, i] / gamma_o), -1, 1).
    W_hat = gamma[:, None] * q.
    """
    weights = jnp.asarray(W)
    gamma = jnp.mean(jnp.abs(weights), axis=-1)
    safe = jnp.where(gamma == 0, 1.0, gamma)
    scaled = weights / safe[:, None]
    q = jnp.clip(jnp.round(scaled), -1, 1)
    q = jnp.where(gamma[:, None] == 0, 0.0, q)
    w_hat = gamma[:, None] * q
    return TernaryCandidates(q=q, gamma=gamma, W_hat=w_hat)


def classify_groups(q: jnp.ndarray, group_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Non-overlapping input groups. Leftover inputs are dropped.

    A group is pure when every ternary value in the slice is identical.
    Returns (is_pure, h) with shape (out_channels, n_groups). h is the
    common value on pure groups and the first entry on mixed groups.
    """
    if group_size < 1:
        raise ValueError("group_size must be positive")
    codes = jnp.asarray(q)
    n_in = codes.shape[-1]
    n_groups = n_in // group_size
    if n_groups == 0:
        empty = codes[:, :0]
        return empty.astype(bool), empty
    grouped = codes[:, : n_groups * group_size].reshape(
        codes.shape[0], n_groups, group_size
    )
    h = grouped[:, :, 0]
    is_pure = jnp.all(grouped == h[:, :, None], axis=-1)
    return is_pure, h


def soft_ternary_probs(
    W: jnp.ndarray,
    gamma: jnp.ndarray | None = None,
    kappa: float = KAPPA_DEFAULT,
) -> jnp.ndarray:
    """Soft assignment p(c) over c in {-1, 0, +1}, last axis in that order."""
    weights = jnp.asarray(W)
    if gamma is None:
        scale = jnp.mean(jnp.abs(weights), axis=-1)
    else:
        scale = jnp.asarray(gamma)
    safe = jnp.where(scale == 0, 1.0, scale)
    u = jnp.where(scale[:, None] == 0, 0.0, weights / safe[:, None])
    s_minus = jax.nn.sigmoid(kappa * (-u - 0.5))
    s_zero = jax.nn.sigmoid(kappa * (0.5 - jnp.abs(u)))
    s_plus = jax.nn.sigmoid(kappa * (u - 0.5))
    scores = jnp.stack([s_minus, s_zero, s_plus], axis=-1)
    return scores / jnp.sum(scores, axis=-1, keepdims=True)


def homogeneity_loss(p: jnp.ndarray, group_size: int) -> jnp.ndarray:
    """L_hom,g = -1/|g| * log(sum_c prod_{i in g} p_i(c)).

    Leftover inputs are dropped. Returns shape (out_channels, n_groups).
    """
    if group_size < 1:
        raise ValueError("group_size must be positive")
    probs = jnp.asarray(p)
    n_in = probs.shape[1]
    n_groups = n_in // group_size
    if n_groups == 0:
        return jnp.zeros((probs.shape[0], 0), dtype=probs.dtype)
    grouped = probs[:, : n_groups * group_size, :].reshape(
        probs.shape[0], n_groups, group_size, probs.shape[-1]
    )
    sum_log = jnp.sum(jnp.log(jnp.clip(grouped, 1e-30, 1.0)), axis=2)
    log_sum_prod = jax.scipy.special.logsumexp(sum_log, axis=-1)
    return -log_sum_prod / group_size


def group_sensitivity(
    W: jnp.ndarray,
    W_hat: jnp.ndarray,
    D: jnp.ndarray,
    group_size: int,
) -> jnp.ndarray:
    """F_g = sum_{i in g} D_i * (W_hat - W)_i^2. Leftover inputs dropped."""
    if group_size < 1:
        raise ValueError("group_size must be positive")
    delta = jnp.asarray(W_hat) - jnp.asarray(W)
    term = jnp.asarray(D) * delta**2
    n_in = term.shape[-1]
    n_groups = n_in // group_size
    if n_groups == 0:
        return jnp.zeros((term.shape[0], 0), dtype=term.dtype)
    grouped = term[:, : n_groups * group_size].reshape(
        term.shape[0], n_groups, group_size
    )
    return jnp.sum(grouped, axis=-1)


def assignment_weights(is_pure: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
    """Pure groups get a_g = 1.

    Mixed groups are standardized on their own F_g (mean and std over mixed
    groups only; std 0 yields z = 0) and then a_g = sigmoid(-z).
    """
    pure = jnp.asarray(is_pure).astype(bool)
    sens = jnp.asarray(F)
    if pure.shape != sens.shape:
        raise ValueError("is_pure and F must have the same shape")
    flat_pure = pure.reshape(-1)
    flat_f = sens.reshape(-1)
    weights = jnp.ones_like(flat_f)
    mixed = ~flat_pure
    n_mixed = int(jnp.sum(mixed))
    if n_mixed == 0:
        return weights.reshape(sens.shape)
    mixed_f = flat_f[mixed]
    std = jnp.std(mixed_f)
    centered = mixed_f - jnp.mean(mixed_f)
    z = jnp.where(std == 0, 0.0, centered / std)
    weights = weights.at[jnp.where(mixed)[0]].set(jax.nn.sigmoid(-z))
    return weights.reshape(sens.shape)


def training_objective(
    task_loss: jnp.ndarray,
    a: jnp.ndarray,
    hom_loss: jnp.ndarray,
    lambda_group: float,
) -> jnp.ndarray:
    """L = L_task + lambda_group / n_groups * sum_g a_g * L_hom,g."""
    coeff = jnp.asarray(a)
    losses = jnp.asarray(hom_loss)
    n_groups = coeff.size
    if n_groups == 0:
        return jnp.asarray(task_loss)
    penalty = jnp.sum(coeff * losses) / n_groups
    return jnp.asarray(task_loss) + lambda_group * penalty


def group_routing_loss(
    W: jnp.ndarray,
    D: jnp.ndarray,
    group_size: int,
    task_loss: jnp.ndarray,
    lambda_group: float,
    kappa: float = KAPPA_DEFAULT,
) -> jnp.ndarray:
    """Full training objective for one packing pool."""
    candidates = ternary_quantize(W)
    is_pure, _h = classify_groups(candidates.q, group_size)
    probs = soft_ternary_probs(W, candidates.gamma, kappa=kappa)
    hom = homogeneity_loss(probs, group_size)
    sens = group_sensitivity(W, candidates.W_hat, D, group_size)
    weights = assignment_weights(is_pure, sens)
    return training_objective(task_loss, weights, hom, lambda_group)


def _signed_cap(n_groups: int, rho_p_max: float) -> int:
    """Largest signed count whose fraction is at most 1 - rho_p_max."""
    limit = (1.0 - float(rho_p_max)) * n_groups
    cap = int(math.floor(limit + 1e-9))
    return max(0, min(n_groups, cap))


def protection_mask(
    is_pure: jnp.ndarray,
    F: jnp.ndarray,
    rho_p_max: float,
) -> jnp.ndarray:
    """Protect highest-F pure groups until the signed-route fraction fits.

    Signed-route fraction is limited to 1 - rho_p_max. If the pure fraction
    does not exceed that limit, protect none. One pool: every group.
    """
    pure = jnp.asarray(is_pure).astype(bool)
    sens = jnp.asarray(F)
    if pure.shape != sens.shape:
        raise ValueError("is_pure and F must have the same shape")
    flat_pure = pure.reshape(-1)
    flat_f = sens.reshape(-1)
    n_groups = int(flat_pure.shape[0])
    n_pure = int(jnp.sum(flat_pure))
    protected = jnp.zeros((n_groups,), dtype=bool)
    if n_groups == 0 or n_pure == 0:
        return protected.reshape(pure.shape)
    cap = _signed_cap(n_groups, rho_p_max)
    if n_pure <= cap:
        return protected.reshape(pure.shape)
    n_protect = n_pure - cap
    pure_idx = jnp.where(flat_pure)[0]
    pure_f = flat_f[pure_idx]
    order = jnp.argsort(-pure_f)
    chosen = pure_idx[order[:n_protect]]
    protected = protected.at[chosen].set(True)
    return protected.reshape(pure.shape)


def forward_weights(
    W: jnp.ndarray,
    W_hat: jnp.ndarray,
    is_pure: jnp.ndarray,
    protected: jnp.ndarray,
    group_size: int,
) -> jnp.ndarray:
    """Unprotected pure groups use W_hat. Mixed, protected, and leftovers use W."""
    if group_size < 1:
        raise ValueError("group_size must be positive")
    weights = jnp.asarray(W)
    hat = jnp.asarray(W_hat)
    pure = jnp.asarray(is_pure).astype(bool)
    guard = jnp.asarray(protected).astype(bool)
    if pure.shape != guard.shape:
        raise ValueError("is_pure and protected must have the same shape")
    use_hat = pure & ~guard
    n_in = weights.shape[-1]
    n_groups = n_in // group_size
    mask = jnp.zeros(weights.shape, dtype=bool)
    if n_groups:
        expanded = jnp.repeat(use_hat, group_size, axis=-1)
        mask = mask.at[:, : n_groups * group_size].set(expanded)
    return jnp.where(mask, hat, weights)
