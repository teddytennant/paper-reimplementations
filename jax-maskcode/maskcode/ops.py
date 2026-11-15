"""Numeric primitives shared by the soft syndrome and the BP decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Smallest magnitude a tanh(mu/2) factor is allowed to take before its log is
# formed. Chosen well above float32 denormals so log() stays finite.
TINY = 1e-30

# Remark 1 of the paper: arctanh is replaced by arctanh((1 - 2*eps)*x) so the
# output is bounded even when the argument reaches +-1. The paper picks
# eps = 1e-7 by grid search over 1e-3 ... 1e-9.
DEFAULT_EPS = 1e-7


def stable_arctanh(x: jax.Array, eps: float = DEFAULT_EPS) -> jax.Array:
    """arctanh((1 - 2*eps) * x), the stabilization from Remark 1.

    At |x| = 1 this returns +-arctanh(1 - 2*eps), roughly +-16.1 for the
    default eps, rather than +-inf. The derivative is still (1-2eps)/(1-((1-2eps)x)^2),
    which reaches about 2.5e6 at the endpoints. That is finite but large, and it
    is exactly the term the paper blames for gradient explosion when BP is
    unrolled.
    """
    return jnp.arctanh((1.0 - 2.0 * eps) * jnp.clip(x, -1.0, 1.0))


def _sign_log(t: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split t into a sign and a log-magnitude, neutralising masked-out entries.

    Masked-out positions get sign +1 and log-magnitude 0, so they contribute a
    factor of exactly 1 to any product taken over the mask.
    """
    sign = jnp.where(t >= 0, 1.0, -1.0)
    sign = jnp.where(mask, sign, 1.0)
    logmag = jnp.log(jnp.clip(jnp.abs(t), TINY, 1.0))
    logmag = jnp.where(mask, logmag, 0.0)
    return sign, logmag


def masked_prod(t: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    """Product of `t` over the entries selected by `mask` along `axis`.

    Done as sign(prod) * exp(sum log|t|) rather than jnp.prod so that a long
    product of factors in (-1, 1) does not underflow to zero. The gradient is
    exact away from t = 0: d/dt_i of the product is (product / t_i), which is
    what the log form produces.
    """
    sign, logmag = _sign_log(t, mask)
    return jnp.prod(sign, axis=axis) * jnp.exp(jnp.sum(logmag, axis=axis))


def masked_prod_excluding(t: jax.Array, mask: jax.Array, axis: int) -> jax.Array:
    """Leave-one-out product along `axis`, restricted to `mask`.

    Entry [.., i, ..] is the product of every masked factor along `axis` except
    factor i. This is the "product over N(j) \\ {i}" that both the check-node
    update (4) and the variable-node update (3) need. Computed by subtracting
    one term from the log-sum instead of dividing, so a factor of zero does not
    blow the whole row up.
    """
    sign, logmag = _sign_log(t, mask)
    total_sign = jnp.expand_dims(jnp.prod(sign, axis=axis), axis)
    total_log = jnp.expand_dims(jnp.sum(logmag, axis=axis), axis)
    # sign is +-1, so dividing out one factor is the same as multiplying by it.
    excl_sign = total_sign * sign
    excl_log = total_log - logmag
    out = excl_sign * jnp.exp(excl_log)
    return jnp.where(mask, out, 0.0)


def llr_from_erf(
    y: jax.Array, sigma: jax.Array | float, eps: float = DEFAULT_EPS
) -> jax.Array:
    """Equation (18): the paper's per-bit LLR of a received symbol.

    Implements

        r_i = log( (1 + erf(y_i / (sqrt(2) sigma))) / (1 - erf(y_i / (sqrt(2) sigma))) )

    with the paper's sign convention, r_i = log( Pr(c_i=1|y_i) / Pr(c_i=0|y_i) ),
    so r_i > 0 means bit 1 is the more likely value. Note this is the logit of
    the Gaussian CDF, log(Phi(y/sigma) / (1 - Phi(y/sigma))), not the textbook
    matched-filter LLR 2y/sigma^2. The two differ because the transmitted
    amplitude |o_1| is itself produced by the network and is not known to be 1,
    so the paper scores only the reliability of the sign.

    The encoder passes sigma_eff = sqrt(sigma_ff^2 + sigma_fb^2) because it sees
    y-tilde. The decoder passes sigma_ff because it sees y directly (eq. 28).

    Written as 2 * arctanh(erf(.)) to avoid log(0) when |erf| rounds to 1; the
    two forms are algebraically identical since log((1+u)/(1-u)) = 2 arctanh(u).
    The same eps stabilization is reused, which caps |r| at about 16.1 for the
    default eps. Equation (18) itself is unbounded; the paper does not say how
    it handles that, and saturating is the only option in float32.
    """
    u = jax.scipy.special.erf(y / (jnp.sqrt(2.0) * sigma))
    return 2.0 * stable_arctanh(u, eps)
