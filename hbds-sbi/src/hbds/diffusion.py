"""Variance-preserving schedule and the compositional diffusion coefficient.

Appendix A.3 prescribes a forward process with ordinary VP mean attenuation and a
narrower accumulated variance V_n(t) = (1 - alpha_bar(t)) / kappa_n(t). The
diffusion coefficient that realizes that variance is equation (5), rewritten as
equation (31):

    g_n(t)^2 = beta(t) * [1 + (n - 1) (1 - alpha_bar(t))^2] / kappa_n(t)^2
    kappa_n(t) = 1 + (n - 1) (1 - alpha_bar(t))

alpha_bar is the paper's cumulative squared signal retention,
alpha_bar(t) = exp(-int_0^t beta(s) ds), with alpha_bar(0) = 1 and
alpha_bar'(t) = -beta(t) alpha_bar(t). The signal multiplier used by the VP
kernel is alpha(t) = exp(-0.5 * int_0^t beta) = sqrt(alpha_bar(t)).

A linear schedule beta(t) = beta_min + t * (beta_max - beta_min) is used, with
the task defaults beta_min = 0.1 and beta_max = 20. The paper's Euler-Maruyama
experiments use beta(t) = 0.05 + 19.95 t; the identities do not depend on that
particular pair, only on alpha_bar = exp(-int beta).

The algebraic form beta * (1 - alpha^2) / (1 - alpha^(2n)) does not tend to
1/n as alpha -> 0, so it is not equation (5). This module implements equation (5).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

BETA_MIN = 0.1
BETA_MAX = 20.0


def beta_schedule(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Linear VP noise rate beta(t) on t in [0, 1]."""
    t = jnp.asarray(t, dtype=jnp.float64)
    return beta_min + t * (beta_max - beta_min)


def integrated_beta(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """int_0^t beta(s) ds for the linear schedule."""
    t = jnp.asarray(t, dtype=jnp.float64)
    return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2


def alpha_bar(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Cumulative squared signal retention. Paper's alpha(t)."""
    return jnp.exp(-integrated_beta(t, beta_min, beta_max))


def signal_scale(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """VP signal multiplier alpha(t) = exp(-0.5 * int_0^t beta)."""
    return jnp.exp(-0.5 * integrated_beta(t, beta_min, beta_max))


def kappa_n(t, n, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Compositional precision factor kappa^{(n)}(t), equation (27)."""
    a = alpha_bar(t, beta_min, beta_max)
    return 1.0 + (n - 1) * (1.0 - a)


def marginal_variance(t, n, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Prescribed noising variance V_n(t), equation (26)."""
    a = alpha_bar(t, beta_min, beta_max)
    return (1.0 - a) / kappa_n(t, n, beta_min, beta_max)


def g_n_squared(t, n, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Diffusion coefficient g_n(t)^2 from equations (5) and (31).

    Satisfies g_1(t)^2 = beta(t), g_n(t)^2 <= beta(t), and
    g_n(t)^2 / beta(t) -> 1/n as alpha_bar(t) -> 0.
    """
    a = alpha_bar(t, beta_min, beta_max)
    num = 1.0 + (n - 1) * (1.0 - a) ** 2
    den = kappa_n(t, n, beta_min, beta_max) ** 2
    return beta_schedule(t, beta_min, beta_max) * num / den


def g_ratio_from_alpha(alpha, n):
    """Closed g_n(t)^2 / beta(t) as a function of paper alpha (squared retention).

    The high-noise limit alpha -> 0 is exactly 1/n.
    """
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    num = 1.0 + (n - 1) * (1.0 - alpha) ** 2
    den = (1.0 + (n - 1) * (1.0 - alpha)) ** 2
    return num / den


def diffusion_weight(t, n=1, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """c(t) = g_n(t)^2 / beta(t), in (1/n, 1] for t in (0, 1]."""
    return g_n_squared(t, n, beta_min, beta_max) / beta_schedule(t, beta_min, beta_max)


def _trapezoid(y, x):
    y = jnp.asarray(y, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    dx = jnp.diff(x)
    mid = 0.5 * (y[:-1] + y[1:])
    return jnp.sum(dx * mid)


def marginal_variance_quadrature(t, n, n_grid=512, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Quadrature of the variance integral implied by Appendix A.3.

    The forward variance ODE is V' = -beta(t) V + g_n(t)^2, V(0) = 0.
    The integrating factor is alpha_bar, so

        V(t) = int_0^t (alpha_bar(t) / alpha_bar(s)) g_n(s)^2 ds.

    This is the integral form of equations (28) and (26). It is compared to
    the closed form V_n(t) on a fixed grid.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    s = jnp.linspace(0.0, t, n_grid)
    ab_t = alpha_bar(t, beta_min, beta_max)
    ab_s = alpha_bar(s, beta_min, beta_max)
    integrand = (ab_t / ab_s) * g_n_squared(s, n, beta_min, beta_max)
    return _trapezoid(integrand, s)


def g_n_squared_from_variance_ode(t, n, h=1e-3, n_grid=512, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Recover g_n^2 from the variance ODE by differentiating a quadrature of V.

    Equation (28): g_n(t)^2 = V'(t) + beta(t) V(t), with V from the integral form.
    """
    vp = marginal_variance_quadrature(t + h, n, n_grid, beta_min, beta_max)
    vm = marginal_variance_quadrature(t - h, n, n_grid, beta_min, beta_max)
    v = marginal_variance_quadrature(t, n, n_grid, beta_min, beta_max)
    vdot = (vp - vm) / (2.0 * h)
    return vdot + beta_schedule(t, beta_min, beta_max) * v


def reverse_drift(state, t, score, n, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Reverse-time sampling drift, equation (37).

    b_t = beta(t) / 2 * state + g_n(t)^2 * score.
    Time runs backward: a positive step of this drift decreases diffusion time.
    """
    bt = beta_schedule(t, beta_min, beta_max)
    g2 = g_n_squared(t, n, beta_min, beta_max)
    return 0.5 * bt * state + g2 * score


def reverse_sde_step(state, t, dt, score, n, key, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """One Euler-Maruyama step of the reverse SDE toward smaller time.

    dt > 0 is the backward step, from diffusion time t to t - dt:

        d theta = [0.5 * beta(t) * theta + g_n(t)^2 * s] dt + g_n(t) dW_bar

    This is the paper's sampler (the full reverse SDE in equation (24), written
    in the reverse-time direction of equation (37)). No final Tweedie correction.
    """
    g2 = g_n_squared(t, n, beta_min, beta_max)
    drift = reverse_drift(state, t, score, n, beta_min, beta_max)
    noise = jax.random.normal(key, jnp.shape(state), dtype=jnp.float64)
    return state + drift * dt + jnp.sqrt(g2 * dt) * noise


def probability_flow_step(state, t, dt, score, n, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Probability-flow ODE step (extra). Half the score coefficient, no noise.

    In forward time the ODE drift is f * state - 0.5 * g^2 * score. Integrating
    toward decreasing time flips the sign and yields

        0.5 * beta * state + 0.5 * g_n^2 * score.
    """
    bt = beta_schedule(t, beta_min, beta_max)
    g2 = g_n_squared(t, n, beta_min, beta_max)
    drift = 0.5 * bt * state + 0.5 * g2 * score
    return state + drift * dt


def reverse_sde_sample(x0, score_fn, n, n_steps, key, t_start=1.0, t_end=1e-3,
                       beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Integrate the reverse SDE from t_start down to t_end.

    score_fn(state, t) must accept a batch state of the same shape as x0.
    """
    ts = jnp.linspace(t_start, t_end, n_steps + 1)

    def step(carry, i):
        x, key = carry
        key, kn = jax.random.split(key)
        t = ts[i]
        dt = ts[i] - ts[i + 1]
        score = score_fn(x, t)
        x = reverse_sde_step(x, t, dt, score, n, kn, beta_min, beta_max)
        return (x, key), None

    (x, _), _ = jax.lax.scan(step, (x0, key), jnp.arange(n_steps))
    return x
