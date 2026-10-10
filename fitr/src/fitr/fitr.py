"""FITR minimization loop: Fisher-diagonal trust region around the incumbent."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

from fitr.fisher import pullback_fisher
from fitr.gp import expected_improvement, fit_mll, posterior, standardize
from fitr.trust import KAPPA_INIT, box, sides, step_length


def fitr_minimize(
    f,
    d,
    n_init=4,
    n_steps=8,
    seed=0,
    n_candidates=64,
    s_mc=8,
    fit_steps=20,
):
    """Minimize f: [0, 1]^d -> R inside a pullback-Fisher trust region.

    The trust-region diagonal is the closed-form pullback of the raw-scale
    GP posterior at the incumbent, so the loop does not depend on Monte Carlo
    noise. s_mc is accepted for API compatibility with the MC estimator and is
    not used for side lengths.

    Returns a dict with x (n, d), y (n,), best_y, kappa, and restarts.
    """
    del s_mc  # closed-form pullback is used so the loop stays deterministic
    rng = np.random.default_rng(seed)
    x_obs = rng.uniform(0.0, 1.0, size=(n_init, d))
    y_obs = np.array([float(f(x_obs[i])) for i in range(n_init)], dtype=np.float64)

    kappa = KAPPA_INIT
    succ = 0
    fail = 0
    restarts = 0

    for step in range(n_steps):
        y_std, y_mean, y_scale = standardize(jnp.asarray(y_obs))
        y_mean = float(y_mean)
        y_scale = float(y_scale)
        x_train = jnp.asarray(x_obs)
        y_train = jnp.asarray(y_std)

        if fit_steps == 0:
            lengthscales = jnp.full((d,), 0.5)
            signal_var = jnp.asarray(1.0)
            noise_var = jnp.asarray(0.1)
        else:
            fitted = fit_mll(
                x_train,
                y_train,
                steps=fit_steps,
                lr=0.05,
                seed=seed + step,
            )
            lengthscales = fitted["lengthscales"]
            signal_var = fitted["signal_var"]
            noise_var = fitted["noise_var"]

        # Incumbent on the raw objective. A restart recenters here next step.
        best_idx = int(np.argmin(y_obs))
        center = jnp.asarray(x_obs[best_idx])

        def mu_and_v(x, x_train=x_train, y_train=y_train, lengthscales=lengthscales,
                     signal_var=signal_var, noise_var=noise_var, y_mean=y_mean,
                     y_scale=y_scale):
            mu_std, var_std = posterior(
                x_train, y_train, x, lengthscales, signal_var, noise_var
            )
            return mu_std * y_scale + y_mean, var_std * (y_scale ** 2)

        metric = pullback_fisher(mu_and_v, center)
        g_diag = jnp.maximum(jnp.diag(metric), 0.0)
        side = sides(g_diag, kappa)
        lo, hi = box(center, side)
        lo_np = np.asarray(lo, dtype=np.float64)
        hi_np = np.asarray(hi, dtype=np.float64)

        unit = rng.uniform(0.0, 1.0, size=(n_candidates, d))
        candidates = np.clip(lo_np + unit * (hi_np - lo_np), 0.0, 1.0)

        mu, var = posterior(
            x_train,
            y_train,
            jnp.asarray(candidates),
            lengthscales,
            signal_var,
            noise_var,
        )
        y_best_std = jnp.min(y_train)
        scores = np.asarray(expected_improvement(mu, var, y_best_std))
        pick = int(np.nanargmax(scores))
        x_new = candidates[pick]
        y_new = float(f(x_new))

        improved = y_new < float(np.min(y_obs))
        x_obs = np.vstack([x_obs, x_new])
        y_obs = np.append(y_obs, y_new)

        kappa, succ, fail, restarted = step_length(kappa, improved, succ, fail, d)
        if restarted:
            restarts += 1

    return {
        "x": x_obs,
        "y": y_obs,
        "best_y": float(np.min(y_obs)),
        "kappa": float(kappa),
        "restarts": int(restarts),
    }
