"""Deep-unfolded likelihood denoiser, paper eq. (7).

s'_{k+1} = Prox_{γ_k g_x^η}( s'_k - γ_k ∇ f_y(s'_k) )
D^L_θ(x, η; y) ≜ s'_K,  s'_0 ≜ x.

The MAP proximal-gradient iteration is unfolded; MSE training targets
the conditional-mean denoiser E[S' | S'+ηN=x, Y'=y] under the
surrogate model (4): S' ~ N(0, τ_0² I), Y' ~ p_{Y|S}(·|S').
g_x^η is coordinate-wise separable, so the proximal is a small MLP
applied independently at each coordinate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgs_unfold.phase import grad_fy


def analytic_prox_g(v, x, eta, tau_0, gamma):
    """Prox_{γ g_x^η}(v) for S' ~ N(0, τ_0² I), x = S' + η N.

    g_x^η(s') = ||s'||²/(2τ_0²) + ||x-s'||²/(2η²)  (up to constants).
    Prox is linear: (v + (γ/η²) x) / (1 + γ/τ_0² + γ/η²).
    """
    eta2 = eta**2
    tau0_2 = tau_0**2
    num = v + (gamma / eta2) * x
    den = 1.0 + gamma / tau0_2 + gamma / eta2
    return num / den


def analytic_unfold(x, eta, y, A, tau_0, meas_std, gammas):
    """Eq. (7) with the closed-form proximal (no learned MLP)."""
    s = x
    for gamma in gammas:
        u = s - gamma * grad_fy(s, y, A, meas_std)
        s = analytic_prox_g(u, x, eta, tau_0, gamma)
    return s


def init_mlp(key, d_in=3, hidden=8, n_hidden=2, d_out=1):
    """He-init SiLU MLP. Input (u_i, x_i, η) per coordinate."""
    sizes = [d_in] + [hidden] * n_hidden + [d_out]
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        w = jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din)
        b = jnp.zeros((dout,))
        params.append((w, b))
    return params


def mlp_apply(params, h):
    last = len(params) - 1
    for i, (w, b) in enumerate(params):
        h = h @ w + b
        if i < last:
            h = jax.nn.silu(h)
    return h


def componentwise_prox_mlp(params, u, x, eta):
    """Replace Prox_{γ g} by a coordinate-wise MLP (paper Sec. III-B)."""
    eta_col = jnp.broadcast_to(jnp.asarray(eta, dtype=u.dtype), u.shape)
    h = jnp.stack([u, x, eta_col], axis=-1)
    return mlp_apply(params, h)[..., 0]


def init_unfolded(
    key, n_layers=2, hidden=8, n_hidden=2, gamma0=0.05
):
    """Pytree params: log_gamma (K,) and a list of K coordinate-wise MLPs.

    γ_k = softplus(log_gamma_k) > 0.
    """
    keys = jax.random.split(key, n_layers)
    # softplus(log_gamma) ≈ gamma0
    log_gamma = jnp.full((n_layers,), jnp.log(jnp.expm1(gamma0)))
    mlps = [init_mlp(k, hidden=hidden, n_hidden=n_hidden) for k in keys]
    return {"log_gamma": log_gamma, "mlps": mlps}


def gammas_from_params(params):
    return jax.nn.softplus(params["log_gamma"])


def unfolded_denoise(params, x, eta, y, A, tau_0, meas_std):
    """D^L_θ(x, η; y) ≜ s'_K, s'_0 ≜ x. Paper eq. (7) with learned prox."""
    del tau_0  # used by the analytic prox; MLP sees (u, x, η) instead
    s = x
    gammas = gammas_from_params(params)
    for k, mlp in enumerate(params["mlps"]):
        u = s - gammas[k] * grad_fy(s, y, A, meas_std)
        s = componentwise_prox_mlp(mlp, u, x, eta)
    return s


def make_likelihood_denoiser(params, y, A, tau_0, meas_std):
    """Callable D(x, η) closing over y for the L-step ODE."""

    def denoiser(x, eta):
        return unfolded_denoise(params, x, eta, y, A, tau_0, meas_std)

    return denoiser


def sample_surrogate_batch(key, A, tau_0, meas_std, batch, eta_min, eta_max):
    """Synthetic (S', Y', η, N) from surrogate model (4)."""
    n = A.shape[1]
    m = A.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    s = tau_0 * jax.random.normal(k1, (batch, n))
    noise_y = meas_std * jax.random.normal(k2, (batch, m))
    y = jnp.abs(s @ A.T) + noise_y
    log_eta = jax.random.uniform(
        k3,
        (batch,),
        minval=jnp.log(eta_min),
        maxval=jnp.log(eta_max),
    )
    eta = jnp.exp(log_eta)
    nrm = jax.random.normal(k4, (batch, n))
    x = s + eta[:, None] * nrm
    return s, y, eta, x


def mse_denoise_loss(params, s, y, eta, x, A, tau_0, meas_std):
    """Paper training: MSE of D^L_θ(x, η; y) against S'."""

    def pred_one(xi, ei, yi):
        return unfolded_denoise(params, xi, ei, yi, A, tau_0, meas_std)

    pred = jax.vmap(pred_one)(x, eta, y)
    return jnp.mean((pred - s) ** 2)


def train_unfolded(
    key,
    A,
    tau_0,
    meas_std,
    steps=20,
    batch=32,
    lr=1e-2,
    n_layers=2,
    hidden=8,
    eta_min=0.05,
    eta_max=1.0,
):
    """SGD on the surrogate MSE. Returns (params, losses)."""
    k_init, key = jax.random.split(key)
    params = init_unfolded(k_init, n_layers=n_layers, hidden=hidden)
    losses = []

    def loss_fn(p, k):
        s, y, eta, x = sample_surrogate_batch(
            k, A, tau_0, meas_std, batch, eta_min, eta_max
        )
        return mse_denoise_loss(p, s, y, eta, x, A, tau_0, meas_std)

    for _ in range(steps):
        key, k_step = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(params, k_step)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        losses.append(loss)
    return params, jnp.stack(losses)


__all__ = [
    "analytic_prox_g",
    "analytic_unfold",
    "componentwise_prox_mlp",
    "gammas_from_params",
    "init_mlp",
    "init_unfolded",
    "make_likelihood_denoiser",
    "mlp_apply",
    "mse_denoise_loss",
    "sample_surrogate_batch",
    "train_unfolded",
    "unfolded_denoise",
]
