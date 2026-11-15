"""Offline pretraining and online last-layer residual least squares (Eqs 12–22)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_lpinn.model import LPINNConfig, features, fluct_value, mean_value, n_eta, predict
from jax_lpinn.physics import ade_operator, feature_derivs, mean_derivs, scalar_derivs


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _adam_update(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda m: m / (1.0 - b1 ** t), m)
    vhat = jax.tree.map(lambda v: v / (1.0 - b2 ** t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t}


def _dm_terms(pred, pred_t, pred_x, pred_xx, u, ut, ux, uxx, cfg: LPINNConfig):
    return jnp.mean(
        (pred - u) ** 2
        + cfg.lam_t * (pred_t - ut) ** 2
        + cfg.lam_x * (pred_x - ux) ** 2
        + cfg.lam_xx * (pred_xx - uxx) ** 2
    )


def train_mean_derivative_matching(
    mean_params,
    x,
    t,
    u_bar,
    ut_bar,
    ux_bar,
    uxx_bar,
    cfg: LPINNConfig,
    steps: int = 40,
    lr: float = 1e-3,
):
    """Fit ĥ̄ to the sample-mean snapshot (Eqs 12–13)."""

    def loss_fn(params):
        u, ut, ux, uxx = mean_derivs(params, x, t, L=cfg.L, T=cfg.T)
        return _dm_terms(u, ut, ux, uxx, u_bar, ut_bar, ux_bar, uxx_bar, cfg)

    state = _adam_init(mean_params)

    def step(carry, _):
        params, st = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, st = _adam_update(params, grads, st, lr=lr)
        return (params, st), loss

    (mean_params, _), losses = jax.lax.scan(step, (mean_params, state), xs=None, length=steps)
    return mean_params, losses


def train_fluct_derivative_matching(
    hidden_params,
    W_samples,
    x,
    t,
    U_prime,
    Ut_prime,
    Ux_prime,
    Uxx_prime,
    cfg: LPINNConfig,
    steps: int = 40,
    lr: float = 1e-3,
):
    """Shared θ̃ with per-snapshot last-layer coefficients {W^{(i)}} (Eq 14)."""

    def loss_fn(params):
        hidden, W = params["hidden"], params["W"]

        def one(Wi, u, ut, ux, uxx):
            def u_fn(xx, tt):
                return fluct_value(hidden, Wi, xx, tt, L=cfg.L, T=cfg.T)

            pu, put, pux, puxx = scalar_derivs(u_fn, x, t)
            return _dm_terms(pu, put, pux, puxx, u, ut, ux, uxx, cfg)

        return jnp.mean(jax.vmap(one)(W, U_prime, Ut_prime, Ux_prime, Uxx_prime))

    params = {"hidden": hidden_params, "W": W_samples}
    state = _adam_init(params)

    def step(carry, _):
        p, st = carry
        loss, grads = jax.value_and_grad(loss_fn)(p)
        p, st = _adam_update(p, grads, st, lr=lr)
        return (p, st), loss

    (params, _), losses = jax.lax.scan(step, (params, state), xs=None, length=steps)
    return params["hidden"], params["W"], losses


def train_residual_based(
    mean_params,
    hidden_params,
    W_samples,
    x_res,
    t_res,
    x_ic,
    t_ic,
    x_bc0,
    t_bc0,
    x_bcL,
    t_bcL,
    V,
    D,
    cfg: LPINNConfig,
    steps: int = 40,
    lr: float = 1e-3,
    train_mean: bool = True,
):
    """Residual-based offline loss (Eq 15) with per-sample W^{(i)} and μ^{(i)}."""

    def one_sample(hidden, mean, Wi, Vi, Di):
        return _residual_objective(
            mean, hidden, Wi, x_res, t_res, x_ic, t_ic, x_bc0, t_bc0, x_bcL, t_bcL,
            Vi, Di, cfg,
        )

    def loss_fn(bundle):
        mean = bundle["mean"] if train_mean else mean_params
        hidden = bundle["hidden"]
        W = bundle["W"]
        return jnp.mean(jax.vmap(lambda Wi, Vi, Di: one_sample(hidden, mean, Wi, Vi, Di))(W, V, D))

    bundle = {"hidden": hidden_params, "W": W_samples}
    if train_mean:
        bundle["mean"] = mean_params
    state = _adam_init(bundle)

    def step(carry, _):
        p, st = carry
        loss, grads = jax.value_and_grad(loss_fn)(p)
        p, st = _adam_update(p, grads, st, lr=lr)
        return (p, st), loss

    (bundle, _), losses = jax.lax.scan(step, (bundle, state), xs=None, length=steps)
    mean_out = bundle["mean"] if train_mean else mean_params
    return mean_out, bundle["hidden"], bundle["W"], losses


def build_ade_lstsq_system(
    mean_params,
    hidden_params,
    x_res,
    t_res,
    x_ic,
    t_ic,
    x_bc0,
    t_bc0,
    x_bcL,
    t_bcL,
    V,
    D,
    cfg: LPINNConfig,
):
    """Linear residual system A W ≈ b for ADE (Eq 20). L is affine in W.

    Stacks PDE residuals, Dirichlet/Neumann BCs, IC, and √λ I regularization.
    """
    n = n_eta(cfg)

    u_m, ut_m, ux_m, uxx_m = mean_derivs(mean_params, x_res, t_res, L=cfg.L, T=cfg.T)
    psi, psi_t, psi_x, psi_xx = feature_derivs(hidden_params, x_res, t_res, L=cfg.L, T=cfg.T)
    L_mean = ade_operator(ut_m, ux_m, uxx_m, V, D)
    L_psi = ade_operator(psi_t, psi_x, psi_xx, V, D)
    A_res = L_psi
    b_res = -L_mean

    psi_ic = features(hidden_params, x_ic, t_ic, L=cfg.L, T=cfg.T)
    mean_ic = mean_value(mean_params, x_ic, t_ic, L=cfg.L, T=cfg.T)
    sqrt_ic = jnp.sqrt(cfg.lam_ic)
    A_ic = sqrt_ic * psi_ic
    b_ic = sqrt_ic * (-mean_ic)

    psi_bc0 = features(hidden_params, x_bc0, t_bc0, L=cfg.L, T=cfg.T)
    mean_bc0 = mean_value(mean_params, x_bc0, t_bc0, L=cfg.L, T=cfg.T)
    sqrt_bc = jnp.sqrt(cfg.lam_bc)
    A_bc0 = sqrt_bc * psi_bc0
    b_bc0 = sqrt_bc * (cfg.u0 - mean_bc0)

    def ux_mean_one(xx, tt):
        return jax.grad(lambda x_: mean_value(mean_params, x_, tt, L=cfg.L, T=cfg.T))(xx)

    def ux_psi_one(xx, tt):
        return jax.jacfwd(lambda x_: features(hidden_params, x_, tt, L=cfg.L, T=cfg.T))(xx)

    ux_m_L = jax.vmap(ux_mean_one)(x_bcL, t_bcL)
    ux_psi_L = jax.vmap(ux_psi_one)(x_bcL, t_bcL)
    A_bcL = sqrt_bc * ux_psi_L
    b_bcL = sqrt_bc * (-ux_m_L)

    sqrt_lam = jnp.sqrt(cfg.lam)
    A_reg = sqrt_lam * jnp.eye(n)
    b_reg = jnp.zeros((n,))

    A = jnp.concatenate([A_res, A_ic, A_bc0, A_bcL, A_reg], axis=0)
    b = jnp.concatenate([b_res, b_ic, b_bc0, b_bcL, b_reg], axis=0)
    return A, b


def online_forward_lstsq(
    mean_params,
    hidden_params,
    x_res,
    t_res,
    x_ic,
    t_ic,
    x_bc0,
    t_bc0,
    x_bcL,
    t_bcL,
    V,
    D,
    cfg: LPINNConfig,
):
    """Solve W_{N+1} by regularized linear least squares (Eq 20). Hidden params frozen."""
    A, b = build_ade_lstsq_system(
        mean_params, hidden_params, x_res, t_res, x_ic, t_ic, x_bc0, t_bc0, x_bcL, t_bcL, V, D, cfg
    )
    W = jnp.linalg.lstsq(A, b, rcond=None)[0]
    return W


def online_objective(
    W,
    mean_params,
    hidden_params,
    x_res,
    t_res,
    x_ic,
    t_ic,
    x_bc0,
    t_bc0,
    x_bcL,
    t_bcL,
    V,
    D,
    cfg: LPINNConfig,
):
    """Eq 20: ||R||_2^2 + λ_BC ||R_BC||_2^2 + λ_IC ||R_IC||_2^2 + λ ||W||_2^2."""
    A, b = build_ade_lstsq_system(
        mean_params, hidden_params, x_res, t_res, x_ic, t_ic, x_bc0, t_bc0, x_bcL, t_bcL, V, D, cfg
    )
    r = A @ W - b
    return jnp.sum(r * r)


def _residual_objective(
    mean_params,
    hidden_params,
    W,
    x_res,
    t_res,
    x_ic,
    t_ic,
    x_bc0,
    t_bc0,
    x_bcL,
    t_bcL,
    V,
    D,
    cfg: LPINNConfig,
):
    return online_objective(
        W, mean_params, hidden_params, x_res, t_res, x_ic, t_ic, x_bc0, t_bc0, x_bcL, t_bcL, V, D, cfg
    )


def fit_reconstruction_W(mean_params, hidden_params, x, t, u, cfg: LPINNConfig, lam: float | None = None):
    """Linear least-squares fit of W so that ĥ̄ + ψ W ≈ u (data reconstruction)."""
    if lam is None:
        lam = cfg.lam
    psi = features(hidden_params, x, t, L=cfg.L, T=cfg.T)
    target = u - mean_value(mean_params, x, t, L=cfg.L, T=cfg.T)
    n = psi.shape[-1]
    A = jnp.concatenate([psi, jnp.sqrt(lam) * jnp.eye(n)], axis=0)
    b = jnp.concatenate([target, jnp.zeros((n,))], axis=0)
    return jnp.linalg.lstsq(A, b, rcond=None)[0]


def inverse_ade(
    mean_params,
    hidden_params,
    x_res,
    t_res,
    x_data,
    t_data,
    u_data,
    cfg: LPINNConfig,
    W_init=None,
    logV_init=None,
    logD_init=None,
    steps: int = 80,
    lr: float = 5e-3,
):
    """Jointly estimate W, V, D from residuals + sparse observations (Eq 22).

    IC/BC weights are left at the config values (paper often sets them to 0
    and uses sparse interior data instead). V and D are parameterized as exp.
    """
    n = n_eta(cfg)
    if W_init is None:
        W_init = jnp.zeros((n,))
    if logV_init is None:
        logV_init = jnp.log(jnp.asarray(cfg.V_star))
    if logD_init is None:
        logD_init = jnp.log(jnp.asarray(cfg.D_star))

    params = {"W": W_init, "logV": jnp.asarray(logV_init), "logD": jnp.asarray(logD_init)}

    # Dummy IC/BC points with λ_ic = λ_bc = 0 so Eq 22 uses residuals + data.
    cfg_inv = LPINNConfig(
        hidden_widths=cfg.hidden_widths,
        L=cfg.L,
        T=cfg.T,
        u0=cfg.u0,
        V_star=cfg.V_star,
        D_star=cfg.D_star,
        lam=cfg.lam,
        lam_ic=0.0,
        lam_bc=0.0,
        lam_data=cfg.lam_data,
    )
    x_dummy = jnp.array([0.5 * cfg.L])
    t_dummy = jnp.array([0.5 * cfg.T])

    def loss_fn(p):
        V = jnp.exp(p["logV"])
        D = jnp.exp(p["logD"])
        W = p["W"]
        A, b = build_ade_lstsq_system(
            mean_params,
            hidden_params,
            x_res,
            t_res,
            x_dummy,
            t_dummy,
            x_dummy,
            t_dummy,
            x_dummy,
            t_dummy,
            V,
            D,
            cfg_inv,
        )
        resid = jnp.sum((A @ W - b) ** 2)
        u_pred = predict(mean_params, hidden_params, W, x_data, t_data, L=cfg.L, T=cfg.T)
        data = cfg.lam_data * jnp.sum((u_pred - u_data) ** 2)
        return resid + data

    state = _adam_init(params)

    def step(carry, _):
        p, st = carry
        loss, grads = jax.value_and_grad(loss_fn)(p)
        p, st = _adam_update(p, grads, st, lr=lr)
        return (p, st), loss

    (params, _), losses = jax.lax.scan(step, (params, state), xs=None, length=steps)
    V = jnp.exp(params["logV"])
    D = jnp.exp(params["logD"])
    return params["W"], V, D, losses
