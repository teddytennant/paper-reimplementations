"""Single-component EM drivers.

`fit_matnorm` runs the loop of Section 3: E-step (exact, partial or plug-in),
then the M-step (10) to (12), and records the observed-data log-likelihood and
the variational bound at every iteration so the monotonicity of each can be
checked directly.
"""

import jax.numpy as jnp

from .estep import (
    exact_estep,
    partial_estep,
    plugin_estep,
    init_estep_state,
    observed_loglik,
    free_energy,
)
from .mstep import mstep
from .params import MatNormParams


def init_params(Y, mask, jitter=0.0):
    """A cheap start: mean of the observed cells, identity factors.

    sigma^2 starts at the variance of the observed cells, which is the right
    scale because |S1| = |S2| = 1.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    obs = ~mask
    w = obs.astype(jnp.float64)
    M = jnp.sum(jnp.where(obs, Y, 0.0), axis=0) / jnp.maximum(jnp.sum(w, axis=0), 1.0)
    resid = jnp.where(obs, Y - M, 0.0)
    sigma2 = jnp.sum(resid**2) / jnp.maximum(jnp.sum(w), 1.0)
    p, q = M.shape
    S1 = jnp.eye(p, dtype=jnp.float64) + jitter * jnp.ones((p, p)) * (1 - jnp.eye(p))
    S2 = jnp.eye(q, dtype=jnp.float64) + jitter * jnp.ones((q, q)) * (1 - jnp.eye(q))
    S1 = S1 / jnp.linalg.det(S1) ** (1.0 / p)
    S2 = S2 / jnp.linalg.det(S2) ** (1.0 / q)
    return MatNormParams(M, S1, S2, jnp.asarray(sigma2))


def _estep(method, Y, mask, params, state, mean_sweeps, cov_sweeps):
    if method == "exact":
        return exact_estep(Y, mask, params)
    if method == "partial":
        return partial_estep(Y, mask, params, state, mean_sweeps, cov_sweeps)
    if method == "plugin":
        return plugin_estep(Y, mask, params, state, mean_sweeps)
    raise ValueError(f"unknown method {method!r}")


def fit_matnorm(
    Y,
    mask,
    method="partial",
    params=None,
    n_iter=50,
    mean_sweeps=1,
    cov_sweeps=1,
    track=True,
):
    """Fit one matrix normal to data with missing cells.

    method: "exact" for the benchmark E-step, "partial" for Section 3,
        "plugin" for conditional means with no covariance correction.

    Returns (params, history) where history["loglik"][t] is the observed-data
    log-likelihood after t M-steps and history["free_energy"][t] the bound of
    the same iterate. Both are evaluated with the exact routines, so the two
    methods are scored on the same scale.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    if params is None:
        params = init_params(Y, mask)
    state = init_estep_state(Y, mask, params)
    hist = {"loglik": [], "free_energy": []}
    if track:
        hist["loglik"].append(float(observed_loglik(Y, mask, params)))
    for _ in range(n_iter):
        state = _estep(method, Y, mask, params, state, mean_sweeps, cov_sweeps)
        if track:
            hist["free_energy"].append(float(free_energy(Y, mask, params, state)))
        params = mstep(state, params)
        if track:
            hist["loglik"].append(float(observed_loglik(Y, mask, params)))
    return params, hist


def missing_rmse(Y_true, Yhat, mask):
    """Imputation error on the deleted cells, the RMSE of Section 3.5."""
    mask = jnp.asarray(mask, dtype=bool)
    d = jnp.where(mask, jnp.asarray(Y_true) - jnp.asarray(Yhat), 0.0)
    return jnp.sqrt(jnp.sum(d**2) / jnp.sum(mask))
