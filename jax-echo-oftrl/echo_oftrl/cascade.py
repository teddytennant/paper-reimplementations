"""The EMA cascade that builds ECHO-OFTRL's high-order optimism vector.

Notation follows Section 3.3 of arXiv:2608.31166. `L` is the backward shift,
`(L z)^(t) = z^(t-1)`, and equation (3.2) sets

    rho = 1 - 1/N,   A = (I - rho L)^-1,   D = I - L,   delta = D A = (I-L)/(I-rho L).

The predictor is ghat = (I - delta^H) g and the prediction error is
e = g - ghat = delta^H g, equation (3.3), with cascade depth H = N.

Two facts from the paper are what make the recursion below correct:

  * I - delta = (1 - rho) L (I - rho L)^-1, a one-pole EMA of strictly past
    values (Section 1.1). So ((I-delta) z)^(t) = (1-rho) sum_{k>=1} rho^(k-1) z^(t-k),
    and ((I-delta) z)^(t+1) = rho ((I-delta) z)^(t) + (1-rho) z^(t).
  * With zero initial states, update (3.6) keeps the invariant
    ema_h^(t) = ((I - delta) delta^(h-1) g)^(t) for every h in [H],
    so summing over h and using (I - delta) sum_h delta^(h-1) = I - delta^H
    gives (3.4), ghat^(t) = sum_h ema_h^(t).

`ema_cascade_step` is lines 3 and 6-8 of Algorithm 1 with the loop over h
replaced by a prefix sum, since e_{h-1}^(t) = g^(t) - sum_{l<h} ema_l^(t).
"""

from __future__ import annotations

import jax.lax as lax
import jax.numpy as jnp


def ema_cascade_predict(ema: jnp.ndarray) -> jnp.ndarray:
    """ghat^(t) = sum_{h=1}^{H} ema_h^(t), equation (3.4).

    `ema` has shape (..., H, m); the sum is over the cascade axis.
    """
    return jnp.sum(ema, axis=-2)


def ema_cascade_step(ema: jnp.ndarray, g: jnp.ndarray, rho: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One round of the cascade, equation (3.6) and Algorithm 1 lines 6-8.

    Args:
      ema: ema^(t), shape (..., H, m).
      g: the observed g^(t), shape (..., m).
      rho: 1 - 1/N in the paper.

    Returns:
      (ema^(t+1), e^(t)) where e^(t) = e_H^(t) = (delta^H g)^(t) is the
      prediction error of equation (3.3).

    Line 8 uses the pre-update ema^(t), not the value written by line 7; the
    superscripts in Algorithm 1 keep them apart and so does this function.
    """
    # exclusive prefix sums along the cascade axis: prefix[h] = sum_{l<h} ema_l
    prefix = jnp.cumsum(ema, axis=-2) - ema
    # e_prev[h] = e_{h-1}^(t) = g^(t) - sum_{l<h} ema_l^(t), with e_0 = g
    e_prev = jnp.expand_dims(g, -2) - prefix
    new_ema = rho * ema + (1.0 - rho) * e_prev
    # e_H = e_0 - sum_{h=1}^{H} ema_h
    error = g - jnp.sum(ema, axis=-2)
    return new_ema, error


def run_cascade(gs: jnp.ndarray, rho: float, depth: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the cascade over a whole gradient sequence, for testing and plotting.

    Args:
      gs: shape (T, m), the sequence g^(1), ..., g^(T).
      rho, depth: rho and H.

    Returns:
      emas of shape (T, depth, m) holding ema^(t) as used at round t (before
      that round's update), and ghats of shape (T, m).
    """
    m = gs.shape[-1]
    init = jnp.zeros((depth, m), dtype=gs.dtype)

    def body(ema, g):
        new_ema, _ = ema_cascade_step(ema, g, rho)
        return new_ema, (ema, ema_cascade_predict(ema))

    _, (emas, ghats) = lax.scan(body, init, gs)
    return emas, ghats


# Reference filter algebra. These are deliberately written as literal operator
# applications so the tests can compare the recursion against the definitions
# in (3.2) rather than against a second copy of itself.


def lag(z: jnp.ndarray) -> jnp.ndarray:
    """(L z)^(t) = z^(t-1), with z^(s) = 0 for s <= 0."""
    return jnp.concatenate([jnp.zeros_like(z[:1]), z[:-1]], axis=0)


def apply_A(z: jnp.ndarray, rho: float) -> jnp.ndarray:
    """A = (I - rho L)^-1 = sum_k rho^k L^k."""
    def body(carry, zt):
        carry = zt + rho * carry
        return carry, carry

    _, out = lax.scan(body, jnp.zeros_like(z[0]), z)
    return out


def apply_delta(z: jnp.ndarray, rho: float) -> jnp.ndarray:
    """delta = (I - L)(I - rho L)^-1, equation (3.2)."""
    w = apply_A(z, rho)
    return w - lag(w)


def apply_I_minus_delta(z: jnp.ndarray, rho: float) -> jnp.ndarray:
    """I - delta, computed the long way as z - delta z."""
    return z - apply_delta(z, rho)
