"""Masked continuous-time NELBO, equation 13.

    L = E_{t ~ U(0,1)} E_q [
        (alpha'(t) / (1 - alpha(t)))
        * sum_{ell: z_t^ell == mask} log <x_theta(z_t)_ell, z_0^ell>
    ]

The inner term sums over masked positions (it is not a mean over positions).
Unmasked positions contribute 0. The weight is negative and the log-prob is
negative, so the term is positive; the sign is kept. On the log-linear
schedule the weight equals -1/t.

SUBS (Sahoo et al.): before the softmax the mask logit is set to -inf, so the
denoiser puts zero mass on the mask token. The loss does not train unmasked
positions; the sampler copies those tokens.
"""

from __future__ import annotations

import jax.numpy as jnp

from refinemix.schedule import EPS, nelbo_weight


def subs_logits(logits, mask_id: int):
    """Set the mask-token logit to -inf (zero mass after softmax)."""
    return logits.at[..., int(mask_id)].set(-jnp.inf)


def token_log_probs(logits, mask_id: int):
    """Log-softmax over the vocabulary after the SUBS constraint."""
    return jax_log_softmax(subs_logits(logits, mask_id))


def jax_log_softmax(logits):
    import jax.nn

    return jax.nn.log_softmax(logits, axis=-1)


def masked_nelbo(
    logits,
    z0,
    zt,
    t,
    mask_id: int,
    *,
    eps: float = EPS,
    ideal: bool = False,
    reduction: str = "mean",
):
    """Inner term of equation 13.

    logits: (B, L, V) raw denoiser logits (mask column is overwritten with -inf)
    z0, zt: (B, L) integer tokens
    t: scalar or (B,)
    reduction: 'mean' (batch mean of the per-sequence sums), 'sum', or 'none'
    """
    z0 = jnp.asarray(z0, dtype=jnp.int32)
    zt = jnp.asarray(zt, dtype=jnp.int32)
    log_probs = token_log_probs(logits, mask_id)
    gathered = jnp.take_along_axis(log_probs, z0[..., None], axis=-1).squeeze(-1)
    masked = zt == int(mask_id)
    # Sum, not mean, over masked positions. Unmasked positions add nothing.
    summed = jnp.sum(jnp.where(masked, gathered, 0.0), axis=-1)
    weight = nelbo_weight(t, eps=eps, ideal=ideal).astype(summed.dtype)
    per_seq = weight * summed
    if reduction == "none":
        return per_seq
    if reduction == "sum":
        return jnp.sum(per_seq)
    if reduction == "mean":
        return jnp.mean(per_seq)
    raise ValueError(f"unknown reduction {reduction!r}")
