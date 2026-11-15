"""Algorithm 3.2: Smoothed Picard HMC, plus exact RGO for quadratics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sphmc.kernel import ScoreFn, picard_hmc_step
from sphmc.prox import prox_quadratic
from sphmc.quadrature import quadrature_weights


class SPHMCConfig(NamedTuple):
    h: float
    n_picard: int
    J: int
    eta: float
    gamma: float


def rgo_quadratic(key: Array, y: Array, A: ArrayLike, eta: float) -> Array:
    """Exact sample from R_{η,y} for V(x) = (1/2) x^T A x.

    R_{η,y} is N( (I + η A)^{-1} y,  η (I + η A)^{-1} ).
    """
    A = jnp.asarray(A, dtype=y.dtype)
    d = A.shape[0]
    mean = prox_quadratic(y, A, eta)
    cov = eta * jnp.linalg.inv(jnp.eye(d, dtype=y.dtype) + eta * A)
    chol = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, y.shape, dtype=y.dtype)
    return mean + chol @ z


def ou_refresh(key: Array, p: Array, h: float, gamma: float) -> Array:
    """P ← e^{-γ h} P + sqrt(1 - e^{-2 γ h}) g,  g ~ N(0, I)."""
    decay = jnp.exp(-gamma * h)
    g = jax.random.normal(key, p.shape, dtype=p.dtype)
    return decay * p + jnp.sqrt(1.0 - decay**2) * g


def sphmc_trajectory(
    key: Array,
    x0: Array,
    score_fn: ScoreFn,
    config: SPHMCConfig,
    n_steps: int,
    p0: Array | None = None,
) -> tuple[Array, Array]:
    """Run K steps of Algorithm 3.2 (smoothed chain only; no RGO).

    Returns the final (X, P) in the smoothed phase space.
    """
    quad = quadrature_weights(config.J, config.h)
    key, k_p = jax.random.split(key)
    p = jax.random.normal(k_p, x0.shape, dtype=x0.dtype) if p0 is None else p0

    def step(carry, k):
        x, p = carry
        k_pic, k_ou = jax.random.split(k)
        state = picard_hmc_step(
            k_pic, x, p, score_fn, quad, config.n_picard
        )
        p_new = ou_refresh(k_ou, state.p, config.h, config.gamma)
        return (state.x, p_new), state.x

    keys = jax.random.split(key, n_steps)
    (x, p), _ = jax.lax.scan(step, (x0, p), keys)
    return x, p


def sample_quadratic(
    key: Array,
    A: ArrayLike,
    config: SPHMCConfig,
    n_steps: int,
    n_samples: int,
    xref: ArrayLike | None = None,
    recover_rgo: bool = True,
) -> Array:
    """Independent SPHMC chains for a quadratic target, optional exact RGO."""
    A = jnp.asarray(A)
    d = A.shape[0]
    xref_arr = jnp.zeros(d, dtype=A.dtype) if xref is None else jnp.asarray(xref, dtype=A.dtype)
    score_fn = _quadratic_score(A, config.eta)

    def one_chain(k):
        k_traj, k_rgo = jax.random.split(k)
        x, _p = sphmc_trajectory(k_traj, xref_arr, score_fn, config, n_steps)
        if recover_rgo:
            return rgo_quadratic(k_rgo, x, A, config.eta)
        return x

    keys = jax.random.split(key, n_samples)
    return jax.vmap(one_chain)(keys)


def _quadratic_score(A: Array, eta: float) -> ScoreFn:
    from sphmc.kernel import make_quadratic_score

    return make_quadratic_score(A, eta)
