"""Adjacent-layer cross-modal attention drift, and the mask-layer choice.

The paper defines the drift at layer ``i`` as the KL divergence from layer
``i-1`` to layer ``i``:

    K_i = D_KL(A_i || A_{i-1}) = sum_x p_i(x) log( p_i(x) / p_{i-1}(x) ).

The argument order matters and is easy to flip. The *current* layer is the
first argument, so it is the one that weights the log-ratio, and the *previous*
layer is the reference in the denominator. Read as a coding cost, K_i is the
excess cost of coding layer ``i``'s attention with layer ``i-1``'s code. A layer
that suddenly puts mass where the previous layer put almost none pays a lot;
the reverse move, dropping mass off a token the previous layer liked, is
cheap. That asymmetry is the paper's, not an implementation detail, and
``tests/test_drift.py`` pins the direction against a scipy oracle.

Collecting every consecutive pair gives the ``L-1`` dimensional drift feature

    F_drift = [K_2, K_3, ..., K_L],

which is 31 dimensional for LLaVA-1.5-7B's 32 decoder layers.

Zeros in the reference. ``p_{i-1}(x) = 0`` with ``p_i(x) > 0`` sends the true KL
to infinity, and real head-averaged attention does produce tokens at exactly 0
in float32 after a sharp softmax. ``smooth`` mixes each distribution with the
uniform distribution over the ``N`` visual tokens,

    p~ = (1 - eps) * p + eps / N,

before the divergence. That keeps every entry at least ``eps / N``, bounds K_i
by ``log(N / eps)``, and leaves the KL of two well-separated distributions
almost unchanged at the default ``eps = 1e-9``. Additive-then-renormalize would
do the same job; mixing with the uniform is used because it keeps the result a
convex combination of valid distributions and the bound is easy to state.
Passing ``eps=0.0`` turns smoothing off and reproduces the textbook KL exactly,
which is what the scipy comparison in the tests uses.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["smooth", "kl_divergence", "drift_profile", "select_mask_layer"]


def smooth(p: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Mix a distribution with the uniform one over its last axis.

    Args:
        p: ``[..., N]`` distribution, assumed to sum to 1 on the last axis.
        eps: mixing weight on the uniform distribution. ``0.0`` is a no-op.

    Returns:
        ``(1 - eps) * p + eps / N``.
    """
    p = jnp.asarray(p)
    if eps == 0.0:
        return p
    n = p.shape[-1]
    return (1.0 - eps) * p + eps / n


def kl_divergence(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """``D_KL(p || q)`` over the last axis, in nats.

    ``p`` is the distribution being coded and ``q`` is the reference. For the
    paper's ``K_i = D_KL(A_i || A_{i-1})`` that means ``p = A_i`` (the current
    layer) and ``q = A_{i-1}`` (the previous one).

    Both arguments are smoothed with ``smooth(..., eps)`` first, so a zero in
    ``q`` gives a large but finite term instead of an infinity. Terms where
    ``p(x) == 0`` contribute 0, matching the ``0 log 0 = 0`` convention, and are
    zeroed by a ``where`` guard rather than by the smoothing, so that ``eps=0``
    is the textbook KL and still returns ``+inf`` when ``q`` puts no mass where
    ``p`` does. Only the smoothing makes that case finite.

    Args:
        p: ``[..., N]`` distribution.
        q: ``[..., N]`` reference distribution, broadcastable against ``p``.
        eps: uniform-mixing weight, see ``smooth``.

    Returns:
        ``[...]`` divergence, one scalar per leading index.
    """
    p = smooth(jnp.asarray(p), eps)
    q = smooth(jnp.asarray(q), eps)
    safe_p = jnp.where(p > 0, p, 1.0)
    terms = p * (jnp.log(safe_p) - jnp.log(q))
    return jnp.sum(jnp.where(p > 0, terms, 0.0), axis=-1)


def drift_profile(dists: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """``F_drift = [K_2, ..., K_L]`` from a layer-wise attention sequence.

    Args:
        dists: ``[L, N]`` normalized per-layer attention over visual tokens, as
            returned by ``cadmp.attention.attention_distributions``.
        eps: uniform-mixing weight passed to ``kl_divergence``.

    Returns:
        ``[L - 1]`` array. Element ``j`` (0-based) is ``K_{j+2}``, the drift
        from layer ``j+1`` to layer ``j+2`` in the paper's 1-based numbering.
    """
    dists = jnp.asarray(dists)
    if dists.ndim != 2:
        raise ValueError(f"expected [L, N] distributions, got {dists.shape}")
    if dists.shape[0] < 2:
        raise ValueError("need at least 2 layers to form an adjacent-layer drift")
    return kl_divergence(dists[1:], dists[:-1], eps=eps)


def select_mask_layer(drift: jnp.ndarray) -> int:
    """Pick the masking layer, ``l_mask = i* - 1`` with ``i* = argmax_i K_i``.

    The off-by-one is the whole content of this function, so it is worth being
    explicit. ``drift`` is 0-based over ``L - 1`` entries and ``drift[j]`` is the
    paper's ``K_{j+2}``. So the paper's ``i* = j* + 2`` for ``j* =
    argmax(drift)``, and

        l_mask = i* - 1 = j* + 1   (1-based)  =  j*   (0-based layer index).

    In other words the returned 0-based layer index is just ``argmax(drift)``,
    which lands on the layer *before* the largest transition, the one the paper
    calls "the model's visual allocation before its strongest reorganization".
    The two boundaries are the sanity check: a peak at the first transition
    ``K_2`` gives ``l_mask = 1`` (0-based 0, the first layer), and a peak at the
    last transition ``K_L`` gives ``l_mask = L - 1`` (0-based ``L - 2``), so the
    final layer is never selected.

    Ties go to the lowest index, following ``argmax``.

    Args:
        drift: ``[L - 1]`` drift profile.

    Returns:
        0-based index of the masking layer, in ``[0, L - 2]``.
    """
    return int(jnp.argmax(jnp.asarray(drift)))
