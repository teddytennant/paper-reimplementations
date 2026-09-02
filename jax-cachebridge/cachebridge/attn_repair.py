"""ATTN-REPAIR, section 3.3: calibration weights from causal attention sensitivity.

Eqs 14-16 build raw sensitivities from the receiver's first future query at each
causal boundary; eqs 17-19 shrink them toward uniform under a Kish floor.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

NEG_INF = -1e30


def attention(q, k, v, mask=None):
    """Causal-slice attention. Returns (a, o) with a_ui = softmax(q_u' k_i / sqrt(d))_i.

    q is (..., G, d), k and v are (..., m, d), mask is (..., m) with True = eligible.
    """
    d = q.shape[-1]
    scores = jnp.einsum("...gd,...md->...gm", q, k) / jnp.sqrt(jnp.asarray(d, q.dtype))
    if mask is not None:
        scores = jnp.where(mask[..., None, :], scores, NEG_INF)
    a = jax.nn.softmax(scores, axis=-1)
    if mask is not None:
        a = jnp.where(mask[..., None, :], a, 0.0)
    o = jnp.einsum("...gm,...md->...gd", a, v)
    return a, o


def first_order_delta(q, k, v, dk, dv, mask=None):
    """Eq 14: delta_o_u = sum_i a_ui dv_i + sum_i a_ui (q_u' dk_i / sqrt(d)) (v_i - o_u)."""
    d = q.shape[-1]
    a, o = attention(q, k, v, mask)
    term_v = jnp.einsum("...gm,...md->...gd", a, dv)
    s = jnp.einsum("...gd,...md->...gm", q, dk) / jnp.sqrt(jnp.asarray(d, q.dtype))
    az = a * s
    term_k = jnp.einsum("...gm,...md->...gd", az, v) - az.sum(-1)[..., None] * o
    return term_v + term_k


def raw_sensitivities(q, k, v, mask=None):
    """Squared Jacobian-block traces over the query heads G(h) sharing KV head h.

    Eq 15: r_i^V = sum_u a_ui^2.
    Eq 16: r_i^K = sum_u a_ui^2 ||v_i - o_u||^2 ||q_u||^2 / d.
    Returns (r_K, r_V), each (..., m).
    """
    d = q.shape[-1]
    a, o = attention(q, k, v, mask)
    a2 = a**2
    r_v = a2.sum(-2)
    dist2 = jnp.sum((v[..., None, :, :] - o[..., :, None, :]) ** 2, axis=-1)  # (..., G, m)
    qnorm2 = jnp.sum(q**2, axis=-1)  # (..., G)
    r_k = jnp.sum(a2 * dist2 * qnorm2[..., None], axis=-2) / d
    if mask is not None:
        r_k = jnp.where(mask, r_k, 0.0)
        r_v = jnp.where(mask, r_v, 0.0)
    return r_k, r_v


def _unit_mass(r):
    total = jnp.sum(r, axis=-1, keepdims=True)
    return jnp.where(total > 0, r / jnp.where(total > 0, total, 1.0), 0.0)


def mean_normalize(r):
    """Rescale so n^-1 sum_i r_i = 1, the convention above eq 17."""
    n = r.shape[-1]
    total = jnp.sum(r, axis=-1, keepdims=True)
    return jnp.where(total > 0, n * r / jnp.where(total > 0, total, 1.0), jnp.ones_like(r))


def cv2(r):
    """CV^2(r) = n^-1 sum_i (r_i - 1)^2 for mean-one r, section 3.3."""
    return jnp.mean((r - 1.0) ** 2, axis=-1)


def shrink(r, alpha):
    """w_i(alpha) = 1 + alpha (r_i - 1), eq 17."""
    return 1.0 + alpha[..., None] * (r - 1.0)


def kish_neff(w):
    """Kish effective sample size, eq 18: (sum_i w_i)^2 / sum_i w_i^2."""
    return jnp.sum(w, axis=-1) ** 2 / jnp.sum(w**2, axis=-1)


def floor_tau(n: int, p_local: int) -> int:
    """tau_l = min(n, 2 p_l), the feature-width heuristic of section 3.3."""
    return min(n, 2 * p_local)


def optimal_alpha(r, tau):
    """Eq 19: the largest alpha in [0, 1] with n_eff(w(alpha)) >= tau."""
    n = r.shape[-1]
    c = cv2(r)
    safe = jnp.where(c > 0, c, 1.0)
    bound = jnp.sqrt(jnp.maximum(n / tau - 1.0, 0.0) / safe)
    return jnp.where(c > 0, jnp.minimum(1.0, bound), jnp.ones_like(c))


def shrunk_weights(r, tau):
    """Mean-normalize the raw sensitivities, then apply eqs 17 and 19. Returns (w, alpha)."""
    r = mean_normalize(r)
    alpha = optimal_alpha(r, tau)
    return shrink(r, alpha), alpha


@functools.partial(jax.jit, static_argnames=("group_size",))
def boundary_sensitivities(q, k, v, sample_pos, boundaries, group_size: int):
    """Accumulate eqs 15-16 over causal boundaries with equal mass per boundary.

    q is (Lt, S, T, Hq, d) post-RoPE target queries, k and v are (Lt, S, T, Ht, d)
    receiver-native cache states, sample_pos are the calibration token positions and
    boundaries are the causal prefix boundaries c. Each (sequence, boundary) pair is
    normalized to unit mass per target layer, head and component before summing.
    Returns (r_K, r_V), each (Lt, Ht, S, n_pos).
    """
    lt, s, _, hq, d = q.shape
    ht = k.shape[-2]
    k_s = jnp.take(k, sample_pos, axis=2)  # (Lt, S, n_pos, Ht, d)
    v_s = jnp.take(v, sample_pos, axis=2)
    k_s = jnp.transpose(k_s, (0, 1, 3, 2, 4))  # (Lt, S, Ht, n_pos, d)
    v_s = jnp.transpose(v_s, (0, 1, 3, 2, 4))

    def one_boundary(c):
        q_c = q[:, :, c, :, :].reshape(lt, s, ht, group_size, d)
        mask = (sample_pos <= c)[None, None, None, :]
        r_k, r_v = raw_sensitivities(q_c, k_s, v_s, mask)
        return _unit_mass(r_k), _unit_mass(r_v)

    r_k, r_v = jax.vmap(one_boundary)(boundaries)

    def collapse(r):
        return jnp.transpose(r.sum(0), (0, 2, 1, 3))  # (Lt, S, Ht, n_pos) -> (Lt, Ht, S, n_pos)

    return collapse(r_k), collapse(r_v)
