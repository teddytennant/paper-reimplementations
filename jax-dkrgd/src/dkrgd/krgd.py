"""Local kernel-based robust gradient descent, eq. (4).

    f_{1,D} = 0,
    f_{t+1,D} = f_{t,D} − (η/|D|) Σ_i G'(ξ_{t,D,σ}(z_i)) (f_{t,D}(x_i) − y_i) K_{x_i},

with ξ = (f(x) − y)²/σ². By the representer theorem the iterate stays in
span{K_{x_i}}, so it is stored as coefficients α ∈ R^{|D|} with
f(x) = Σ_i α_i K(x_i, x).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from dkrgd.kernel import kernel_matrix
from dkrgd.losses import C_G, G_PRIME_PLUS_0, G_prime, LossName


class KRGDModel(NamedTuple):
    X: jnp.ndarray
    alpha: jnp.ndarray
    lengthscale: float
    n: int


def step_size_bound(kappa: float = 1.0) -> float:
    """Paper's restriction 0 < η ≤ 1 / (κ² max{G'_+(0), C_G}) (Theorems 1–3)."""
    return 1.0 / (kappa**2 * max(G_PRIME_PLUS_0, C_G))


def fit_krgd(
    X,
    y,
    *,
    n_steps: int,
    eta: float,
    sigma: float,
    loss: LossName = "welsch",
    lengthscale: float = 1.0,
) -> KRGDModel:
    """Run T = n_steps updates of (4) from f_1 = 0; return f_{T+1,D}."""
    X = jnp.asarray(X)
    y = jnp.asarray(y).reshape(-1)
    n = y.shape[0]
    K = kernel_matrix(X, lengthscale=lengthscale)
    alpha0 = jnp.zeros((n,), dtype=K.dtype)

    def body(alpha, _):
        pred = K @ alpha
        resid = pred - y
        xi = (resid / sigma) ** 2
        alpha_next = alpha - (eta / n) * G_prime(xi, loss=loss) * resid
        return alpha_next, None

    alpha, _ = jax.lax.scan(body, alpha0, xs=None, length=int(n_steps))
    return KRGDModel(X=X, alpha=alpha, lengthscale=float(lengthscale), n=int(n))


def predict_krgd(model: KRGDModel, X_new):
    """Evaluate f(x) = Σ_i α_i K(x_i, x)."""
    X_new = jnp.asarray(X_new)
    Kx = kernel_matrix(X_new, model.X, lengthscale=model.lengthscale)
    return Kx @ model.alpha
