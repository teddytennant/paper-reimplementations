"""Two runnable examples. `python -m sme_betel.demo`.

The first is a one-parameter misspecified unnormalized model where the
calibration claim of Section 4 can be checked by simulation. The second is a
small mixed-domain preferential sampling fit in the spirit of Section 3.1.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from . import models
from .asymptotics import sandwich_covariance
from .mixed import box_weight, mixed_estimator, mixed_moment_fn, mixed_objective
from .posterior import log_posterior_fn, metropolis_hastings
from .scorematch import moment_fn, moment_matrix

MIX_SD = (1.0, 3.0)
MIX_P = 0.1
_VAR = (1 - MIX_P) * MIX_SD[0] ** 2 + MIX_P * MIX_SD[1] ** 2


def _mixture(key, shape):
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, shape)
    heavy = jax.random.uniform(k2, shape) < MIX_P
    return z * jnp.where(heavy, MIX_SD[1], MIX_SD[0]) / jnp.sqrt(_VAR)


def misspecified_precision(n: int = 200, reps: int = 80, draws: int = 6000) -> None:
    """Gaussian working model, Gaussian scale mixture truth, theta* = 1.

    Reports the SME-BETEL posterior spread against the sampling spread of the
    score matching estimator, and the frequentist coverage of both the
    SME-BETEL credible interval and the interval you get by reading Jhat as a
    log-likelihood.
    """
    print("=" * 66)
    print("misspecified Gaussian precision, n =", n)
    log_q = models.gaussian_precision_log_q(1)
    g = moment_fn(log_q)
    gmat = moment_matrix(g)
    log_post = log_posterior_fn(gmat, lambda t: -0.5 * jnp.sum(t**2) / 100.0)

    big = _mixture(jax.random.key(0), (400_000,))
    theta_star = 1.0 / float(jnp.mean(big**2))
    v = float(sandwich_covariance(g, big[:, None], jnp.array([theta_star]))[0, 0])
    print(f"  score matching target theta*        {theta_star: .4f}")
    print(f"  sandwich sd at n                    {np.sqrt(v / n): .4f}")

    def one(key):
        x = _mixture(jax.random.fold_in(key, 0), (n,))[:, None]
        hat = 1.0 / jnp.mean(x**2)
        out = metropolis_hastings(
            lambda t: log_post(t, x),
            hat[None],
            jax.random.fold_in(key, 1),
            n_iter=draws,
            burn_in=draws // 3,
            init_scale=np.sqrt(v / n),
        )
        s = out.samples[:, 0]
        lo, hi = jnp.quantile(s, jnp.array([0.025, 0.975]))
        # Reading Jhat as a log-likelihood gives an exactly Gaussian posterior
        # with variance theta_hat / n for this model.
        half = 1.959963984540054 * jnp.sqrt(hat / n)
        return hat, jnp.std(s), lo, hi, hat - half, hat + half

    t0 = time.time()
    hat, sd, lo, hi, nlo, nhi = jax.jit(jax.vmap(one))(
        jax.random.split(jax.random.key(1), reps)
    )
    hat, sd, lo, hi, nlo, nhi = (np.asarray(a) for a in (hat, sd, lo, hi, nlo, nhi))
    print(f"  repeated-sample sd of theta_hat_n   {hat.std(): .4f}")
    print(f"  mean SME-BETEL posterior sd         {sd.mean(): .4f}")
    print(f"  mean naive (Jhat as loglik) sd      {np.mean(np.sqrt(hat / n)): .4f}")
    print(f"  SME-BETEL 95% coverage of theta*    {np.mean((lo <= theta_star) & (hi >= theta_star)): .3f}")
    print(f"  naive 95% coverage of theta*        {np.mean((nlo <= theta_star) & (nhi >= theta_star)): .3f}")
    print(f"  ({reps} replications, {time.time() - t0:.1f}s)")


def preferential_sampling(n: int = 600, draws: int = 12000) -> None:
    """Mixed-domain fit, eq (2.9) with the eq (S1) weight and epsilon = 0.2.

    Locations on [0, 1]^2 with density proportional to exp(xi(s; beta)),
    responses Student t about eta0 + a xi(s) while the working model says
    Gaussian. Same shape as Section 3.1, five parameters instead of 76.
    """
    print("=" * 66)
    print("mixed-domain preferential sampling, n =", n)
    theta_true = jnp.array([1.5, -1.0, 0.8, 0.5, float(np.log(0.25))])
    names = ["beta_1", "beta_2", "a", "eta_0", "log sigma^2"]

    h = box_weight(jnp.zeros(2), jnp.ones(2), 0.2)
    jmd = mixed_objective(models.prefsamp_log_q, h)
    gmd = mixed_moment_fn(models.prefsamp_log_q, h)
    gmat = jax.vmap(gmd, in_axes=(0, 0, None))

    # Pseudo-true value: minimize J_MD on a very large sample from the truth.
    ybig, sbig = models.sample_prefsamp(jax.random.key(2), 200_000, theta_true)
    theta_star = mixed_estimator(jmd, ybig, sbig, jnp.zeros(5))

    y, s = models.sample_prefsamp(jax.random.key(3), n, theta_true)
    theta_hat = mixed_estimator(jmd, y, s, jnp.zeros(5))

    # The sandwich of eq (4.2) needs g as a function of one stacked observation.
    stacked = jnp.concatenate([y, s], axis=1)
    sand = sandwich_covariance(
        lambda z, th: gmd(z[:1], z[1:], th), stacked, theta_hat
    ) / n

    log_post = log_posterior_fn(
        lambda data, th: gmat(data[0], data[1], th),
        lambda t: -0.5 * jnp.sum(t**2) / 25.0,
    )
    t0 = time.time()
    out = jax.jit(
        lambda k: metropolis_hastings(
            lambda t: log_post(t, (y, s)),
            theta_hat,
            k,
            n_iter=draws,
            burn_in=draws // 4,
            step_chol=jnp.linalg.cholesky(sand),
            init_scale=1.0,
        )
    )(jax.random.key(4))
    draws_arr = np.asarray(out.samples)
    print(f"  acceptance {float(out.accept_rate):.2f}, {time.time() - t0:.1f}s")
    print(
        f"  {'':12s} {'truth':>8s} {'theta*':>8s} {'theta_hat':>9s}"
        f" {'post mean':>10s} {'post sd':>8s} {'sandwich':>9s}"
    )
    sand_sd = np.sqrt(np.diag(np.asarray(sand)))
    for j, name in enumerate(names):
        print(
            f"  {name:12s} {float(theta_true[j]):8.3f} {float(theta_star[j]):8.3f}"
            f" {float(theta_hat[j]):9.3f} {draws_arr[:, j].mean():10.3f}"
            f" {draws_arr[:, j].std():8.3f} {sand_sd[j]:9.3f}"
        )
    print("  theta* is the minimizer of J_MD under the truth, theta_hat its estimate")
    print("  at this n. The posterior tracks theta_hat, as Theorem 1 says it should.")


def main() -> None:
    misspecified_precision()
    preferential_sampling()


if __name__ == "__main__":
    main()
