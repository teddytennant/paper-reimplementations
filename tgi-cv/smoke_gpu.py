"""Train an affine map, then compare plain and control-variate TGI-RWM variances."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "0")

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()

import jax
import jax.numpy as jnp

from tgi_cv import (
    affine_apply,
    affine_linear_coefficients,
    beta_from_gamma,
    degree1_modes,
    log_w_affine,
    lower_triangular,
    mode_eigenvalues,
    objective,
    poisson_eta,
    run_tgi_rwm,
    sample_tail,
    train_affine,
)


def target_log_prob(x):
    """Unnormalized banana density, shifted so an affine map has a mean to fit."""
    x0 = x[..., 0]
    x1 = x[..., 1]
    bend = 0.35
    return -0.5 * (x0**2 + (x1 - 0.9 - bend * x0**2) ** 2)


def main():
    cpu = jax.default_backend() == "cpu" or os.environ.get("JAX_PLATFORMS") == "cpu"
    d = 2
    n_steps = 200 if cpu else 400
    batch = 128 if cpu else 256
    n_tail = 32 if cpu else 64
    n_mcmc = 1500 if cpu else 4000
    tail_weight = 1e-3
    c_tail = 3.0
    lr = 0.08
    gamma = 0.35
    weight = jnp.array([1.0, 0.25])

    key = jax.random.PRNGKey(0)
    key, key_train, key_eval, key_chain = jax.random.split(key, 4)
    z_eval = jax.random.normal(key_eval, (2048, d))
    zeta_eval = sample_tail(jax.random.fold_in(key_eval, 1), d, 256)

    params0 = {
        "raw_tril": jnp.zeros((d * (d + 1) // 2,)),
        "b": jnp.zeros((d,)),
    }
    loss_start = objective(params0, z_eval, zeta_eval, target_log_prob, tail_weight, c_tail)

    params, _ = train_affine(
        key_train,
        target_log_prob,
        d,
        n_steps=n_steps,
        batch=batch,
        n_tail=n_tail,
        tail_weight=tail_weight,
        c=c_tail,
        lr=lr,
        params=params0,
    )
    loss_end = objective(params, z_eval, zeta_eval, target_log_prob, tail_weight, c_tail)

    L = lower_triangular(params["raw_tril"], d)
    modes = degree1_modes(d)
    lam = mode_eigenvalues(modes, beta_from_gamma(gamma))
    c_f = affine_linear_coefficients(L, weight)
    eta = poisson_eta(c_f, lam)

    def log_w_fn(z):
        return log_w_affine(z, params, target_log_prob)

    def f_fn(z):
        x, _, _ = affine_apply(z, params)
        return jnp.dot(weight, x)

    z0 = jnp.zeros((d,))
    summands, plain, accept, _ = run_tgi_rwm(
        key_chain, z0, log_w_fn, f_fn, gamma, n_mcmc, modes, eta
    )
    # Drop a short burn-in so the reported variances are on the chain, not z0.
    burn = 50
    var_plain = jnp.var(plain[burn:])
    var_cv = jnp.var(summands[burn:])
    accept_rate = jnp.mean(accept[burn:].astype(jnp.float64))

    payload = {
        "loss_start": float(loss_start),
        "loss_end": float(loss_end),
        "var_plain": float(var_plain),
        "var_cv": float(var_cv),
        "accept_rate": float(accept_rate),
    }
    print(json.dumps(payload))
    if not (payload["loss_end"] < payload["loss_start"]):
        raise SystemExit("loss did not decrease")
    if not (payload["var_cv"] < payload["var_plain"]):
        raise SystemExit("control variate did not reduce variance")


if __name__ == "__main__":
    main()
