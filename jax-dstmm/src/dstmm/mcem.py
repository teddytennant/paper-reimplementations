"""The stochastic / Monte Carlo EM algorithm of Section 3.

One iteration is Algorithm 1:

  1. collapse every path with (5)-(7);
  2. evaluate the pathway responsibilities tau_js of (12) on the log scale, as in
     Section 3.2;
  3. for every observation-path pair draw W_{js,1:M} from (21), the GIG law of
     Proposition 2, or from the inverse-gamma limit (17) on symmetric paths;
  4. with each of those draws held fixed, sample z^(1), ..., z^(h) top down from
     the conditional (23); the same W is reused at every transition, which is
     what keeps the pathway GHST marginal (10) exact;
  5. form the weighted cross-products (24)-(25), pool them over paths that use a
     given local component, and update B = (eta, Lambda, delta) by (26), the
     mixing proportions by (27), and the specific covariances by (28), keeping
     only the diagonal and flooring it;
  6. update the degrees of freedom by solving (29), with E(log W) and E(W^-1)
     taken from (16) and (22) rather than from the Monte Carlo draws;
  7. evaluate (11).

With M = 1 this is stochastic EM, so Section 3.6 says the observed log-likelihood
is not guaranteed to increase at every iteration and a moving-window summary is
what should be monitored.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma

from dstmm.gig import gig_log_moment, gig_moment, sample_gig, sample_inverse_gamma
from dstmm.model import (
    DStMM,
    collapse_paths,
    log_likelihood,
    path_component_mask,
    path_responsibilities,
)

# Section 3.2 evaluates the Student-t limit instead of the Bessel form when A_s is
# numerically negligible; the same threshold picks the inverse-gamma limit (17)
# for the conditional draw.
_A_FLOOR = 1e-12


@dataclasses.dataclass
class MCEMResult:
    """Fitted model plus the log-likelihood trace of equation (11), one entry per iteration."""

    model: DStMM
    log_lik: jnp.ndarray
    responsibilities: jnp.ndarray


def _path_layer_index(model, layer):
    return jnp.array([s[layer - 1] for s in model.paths])


def _draw_w(key, model, y, mu, alpha, sigma, n_mc):
    """Step 3: draw W_{js,m} from (21), falling back to (17) on symmetric paths."""
    n, p = y.shape
    n_paths = mu.shape[0]
    chol = jnp.linalg.cholesky(sigma)
    resid = y[:, None, :] - mu[None, :, :]  # (n, |S|, p)
    sr = jax.scipy.linalg.solve_triangular(
        chol[None], resid[..., None], lower=True
    )[..., 0]
    sa = jax.scipy.linalg.solve_triangular(chol, alpha[..., None], lower=True)[..., 0]
    d_js = jnp.sum(sr * sr, axis=-1)  # equation (14)
    a_s = jnp.sum(sa * sa, axis=-1)  # equation (15)

    nu = model.nu
    psi = jnp.broadcast_to(a_s[None, :, None], (n, n_paths, n_mc))
    chi = jnp.broadcast_to((nu[None, :] + d_js)[:, :, None], (n, n_paths, n_mc))
    lam = jnp.broadcast_to((-0.5 * (nu + p))[None, :, None], (n, n_paths, n_mc))

    skewed = a_s > _A_FLOOR
    k_gig, k_ig = jax.random.split(key)
    w_gig = sample_gig(k_gig, jnp.maximum(psi, _A_FLOOR), chi, lam)
    w_ig = sample_inverse_gamma(k_ig, -lam, 0.5 * chi)
    w = jnp.where(skewed[None, :, None], w_gig, w_ig)
    return w, d_js, a_s


def _analytic_w_moments(model, d_js, a_s, p):
    """E(W^-1) and E(log W) from (16) and (22), with the (18) limit when A_s = 0."""
    nu = model.nu
    psi = jnp.maximum(a_s, _A_FLOOR)[None, :]
    chi = nu[None, :] + d_js
    lam = (-0.5 * (nu + p))[None, :]

    e_inv_gig = gig_moment(-1.0, psi, chi, lam)
    e_log_gig = gig_log_moment(psi, chi, lam)
    e_inv_ig = (nu[None, :] + p) / chi  # equation (18)
    e_log_ig = jnp.log(0.5 * chi) - digamma(-lam)

    skewed = (a_s > _A_FLOOR)[None, :]
    return jnp.where(skewed, e_inv_gig, e_inv_ig), jnp.where(skewed, e_log_gig, e_log_ig)


def _simulate_latents(key, model, y, mu_seq, alpha_seq, sigma_seq, w):
    """Step 4: sample z^(1..h) from (23), reusing the same W down the whole path."""
    h = model.depth
    n, n_paths, n_mc = w.shape
    z = jnp.broadcast_to(y[:, None, None, :], (n, n_paths, n_mc, model.dims[0]))
    latents = [z]
    keys = jax.random.split(key, h)
    for l in range(1, h + 1):
        a = _path_layer_index(model, l)
        lam_p = model.lam[l - 1][a]  # (|S|, r_{l-1}, r_l)
        psi_p = model.psi_diag[l - 1][a]  # (|S|, r_{l-1})
        eta_p = model.eta[l - 1][a]
        dlt_p = model.delta[l - 1][a]
        sig_l = sigma_seq[l]  # (|S|, r_l, r_l)
        mu_l, al_l = mu_seq[l], alpha_seq[l]

        lam_pinv = lam_p / psi_p[:, :, None]  # Psi^-1 Lambda
        prec = jnp.linalg.inv(sig_l) + jnp.einsum("sji,sjk->sik", lam_p, lam_pinv)
        omega = jnp.linalg.inv(prec)  # Omega_s^(l)
        omega_chol = jnp.linalg.cholesky(omega)

        prior_mean = mu_l[None, :, None, :] + w[..., None] * al_l[None, :, None, :]
        t1 = jnp.einsum("sij,nsmj->nsmi", jnp.linalg.inv(sig_l), prior_mean)
        resid = z - eta_p[None, :, None, :] - w[..., None] * dlt_p[None, :, None, :]
        t2 = jnp.einsum("sji,nsmj->nsmi", lam_pinv, resid)
        rho = jnp.einsum("sij,nsmj->nsmi", omega, t1 + t2)

        eps = jax.random.normal(keys[l - 1], rho.shape, dtype=jnp.float64)
        z = rho + jnp.sqrt(w)[..., None] * jnp.einsum("sij,nsmj->nsmi", omega_chol, eps)
        latents.append(z)
    return latents


def _update_layer(model, layer, tau, latents, w, active, psi_floor, ridge):
    """Steps 5: equations (24)-(26) and (28) for one layer."""
    z_prev = latents[layer - 1]
    z_cur = latents[layer]
    n, _, n_mc, _ = z_cur.shape
    inv_w = 1.0 / w

    parts = [jnp.ones(z_cur.shape[:-1] + (1,)), z_cur]
    if active:
        parts.append(w[..., None])
    x = jnp.concatenate(parts, axis=-1)  # x-tilde of Section 3.4

    r_all = jnp.einsum("ns,nsm,nsmi,nsmj->sij", tau, inv_w, z_prev, x) / n_mc
    q_all = jnp.einsum("ns,nsm,nsmi,nsmj->sij", tau, inv_w, x, x) / n_mc
    mask = path_component_mask(model, layer)  # (|S|, K_l)
    r_a = jnp.einsum("sk,sij->kij", mask, r_all)
    q_a = jnp.einsum("sk,sij->kij", mask, q_all)

    q_a = q_a + ridge * jnp.eye(q_a.shape[-1])[None]
    b_a = jnp.linalg.solve(jnp.swapaxes(q_a, -1, -2), jnp.swapaxes(r_a, -1, -2))
    b_a = jnp.swapaxes(b_a, -1, -2)  # (K_l, r_{l-1}, m_l), equation (26)

    r_cur = z_cur.shape[-1]
    eta_new = b_a[:, :, 0]
    lam_new = b_a[:, :, 1 : 1 + r_cur]
    delta_new = b_a[:, :, 1 + r_cur] if active else jnp.zeros_like(eta_new)

    a_idx = _path_layer_index(model, layer)
    e = z_prev - jnp.einsum("sij,nsmj->nsmi", b_a[a_idx], x)
    sq = jnp.einsum("ns,nsm,nsmi->si", tau, inv_w, e * e) / n_mc
    n_a = jnp.einsum("ns,sk->k", tau, mask)
    psi_new = jnp.einsum("sk,si->ki", mask, sq) / n_a[:, None]  # diagonal of (28)
    psi_new = jnp.maximum(psi_new, psi_floor)

    log_pi_new = jnp.log(n_a / n)  # equation (27)
    return log_pi_new, eta_new, lam_new, psi_new, delta_new


def _solve_df(c, nu_lo, nu_hi, n_bisect=60):
    """Bisect the scalar equation (29): log(nu/2) - psi(nu/2) + 1 - c = 0.

    log(x) - digamma(x) is strictly decreasing, so the left side decreases in nu
    and a bracket plus bisection is enough. The root is clamped to the bracket,
    which is the safeguard of Section 3.5.
    """

    def g(nu):
        return jnp.log(0.5 * nu) - digamma(0.5 * nu) + 1.0 - c

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        go = g(mid) > 0.0  # still decreasing, root is above mid
        return jnp.where(go, mid, lo), jnp.where(go, hi, mid)

    lo = jnp.full_like(c, nu_lo)
    hi = jnp.full_like(c, nu_hi)
    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
    return 0.5 * (lo + hi)


@partial(jax.jit, static_argnames=("active_set", "n_mc", "df_mode"))
def mcem_step(
    model,
    y,
    key,
    active_set=(1,),
    n_mc=1,
    df_mode="common",
    psi_floor=1e-6,
    nu_bounds=(2.01, 200.0),
    ridge=1e-8,
):
    """One iteration of Algorithm 1. Returns (new_model, tau, log-likelihood before the step)."""
    n, p = y.shape
    mu_seq, alpha_seq, sigma_seq, _ = collapse_paths(model)
    tau = path_responsibilities(model, y)
    ll = log_likelihood(model, y)

    k_w, k_z = jax.random.split(key)
    w, d_js, a_s = _draw_w(k_w, model, y, mu_seq[0], alpha_seq[0], sigma_seq[0], n_mc)
    latents = _simulate_latents(k_z, model, y, mu_seq, alpha_seq, sigma_seq, w)

    log_pi, eta, lam, psi_diag, delta = [], [], [], [], []
    for l in range(1, model.depth + 1):
        out = _update_layer(
            model, l, tau, latents, w, l in active_set, psi_floor, ridge
        )
        for store, val in zip((log_pi, eta, lam, psi_diag, delta), out):
            store.append(val)

    e_inv, e_log = _analytic_w_moments(model, d_js, a_s, p)
    if df_mode == "common":
        c = jnp.sum(tau * (e_log + e_inv)) / n
        nu_new = jnp.full_like(model.nu, _solve_df(c, *nu_bounds))
    elif df_mode == "path":
        n_s = jnp.sum(tau, axis=0)
        c = jnp.sum(tau * (e_log + e_inv), axis=0) / n_s
        nu_new = _solve_df(c, *nu_bounds)
    else:
        raise ValueError(f"unknown df_mode {df_mode!r}")

    new = model.replace(
        log_pi=tuple(log_pi),
        eta=tuple(eta),
        lam=tuple(lam),
        psi_diag=tuple(psi_diag),
        delta=tuple(delta),
        nu=nu_new,
    )
    return new, tau, ll


def mcem_fit(
    key,
    model,
    y,
    n_iter=50,
    active_set=(1,),
    n_mc=1,
    df_mode="common",
    psi_floor=1e-6,
    nu_bounds=(2.01, 200.0),
    ridge=1e-8,
):
    """Run Algorithm 1 for n_iter iterations.

    The trace holds l(Theta^(t)) of equation (11) evaluated at the parameters
    entering iteration t, plus one final entry at the fitted parameters. Section
    3.6 warns that this is not monotone at M = 1.
    """
    trace = []
    for _ in range(n_iter):
        key, sub = jax.random.split(key)
        model, _, ll = mcem_step(
            model,
            y,
            sub,
            active_set=tuple(active_set),
            n_mc=n_mc,
            df_mode=df_mode,
            psi_floor=psi_floor,
            nu_bounds=nu_bounds,
            ridge=ridge,
        )
        trace.append(ll)
    trace.append(log_likelihood(model, y))
    tau = path_responsibilities(model, y)
    return MCEMResult(model=model, log_lik=jnp.stack(trace), responsibilities=tau)
